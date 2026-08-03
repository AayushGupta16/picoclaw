// PicoClaw - Ultra-lightweight personal AI agent

package agent

import (
	"context"
	"strings"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/logger"
)

// runBackgroundTurns drains self-generated inbound messages one at a time for
// the life of the receive loop. Serializing them here preserves the ordering
// the old inline path had, without holding the goroutine that has to keep
// dequeuing the bus.
func (al *AgentLoop) runBackgroundTurns(ctx context.Context) {
	for {
		msg, ok := al.backgroundTurns.pop()
		if !ok {
			select {
			case <-ctx.Done():
				return
			case <-al.backgroundTurns.signal:
			}
			continue
		}
		if err := al.turns.acquire(ctx, laneBackground); err != nil {
			return
		}
		al.runBackgroundTurn(ctx, msg)
	}
}

func (al *AgentLoop) runBackgroundTurn(ctx context.Context, msg bus.InboundMessage) {
	defer al.turns.release()
	defer func() {
		if r := recover(); r != nil {
			logger.RecoverPanicNoExit(r)
			logger.ErrorCF("agent", "Background turn panicked",
				map[string]any{
					"channel":   msg.Channel,
					"chat_id":   msg.ChatID,
					"sender_id": msg.SenderID,
				})
		}
	}()

	al.processMessageSync(ctx, msg)
}

func (al *AgentLoop) rememberSteeringTarget(target continuationTarget) {
	if strings.TrimSpace(target.SessionKey) == "" {
		return
	}
	al.steeringTargets.Store(target.SessionKey, target)
}

func (al *AgentLoop) steeringTarget(sessionKey string) continuationTarget {
	if value, ok := al.steeringTargets.Load(sessionKey); ok {
		if target, ok := value.(continuationTarget); ok {
			return target
		}
	}
	return continuationTarget{SessionKey: sessionKey}
}

// sweepStrandedSteering starts a turn for any scope holding queued messages
// with no turn to inject them into.
//
// A scope's queue is otherwise only read by the turn that owns the session and
// drained, after that turn ends, by the worker that enqueued the message. When
// neither exists — the owning worker is still waiting on the gate, its
// placeholder was released, or the message arrived after the owning turn
// passed its last poll — the message stays in memory with nothing left to
// consume it. This is the turn-boundary check that makes the queue recover on
// its own instead of stranding a human message until the next restart.
func (al *AgentLoop) sweepStrandedSteering(ctx context.Context) {
	if al.steering == nil {
		return
	}
	for _, sessionKey := range al.steering.pendingScopes() {
		if al.getActiveTurnState(sessionKey) != nil {
			continue
		}
		if _, sweeping := al.sweepingScopes.LoadOrStore(sessionKey, struct{}{}); sweeping {
			continue
		}
		go al.recoverStrandedSteering(ctx, al.steeringTarget(sessionKey))
	}
}

func (al *AgentLoop) recoverStrandedSteering(ctx context.Context, target continuationTarget) {
	defer al.sweepingScopes.Delete(target.SessionKey)

	if err := al.turns.acquire(ctx, laneHuman); err != nil {
		return
	}
	defer al.turns.release()

	logger.InfoCF("agent", "Recovering stranded steering messages",
		map[string]any{
			"session_key": target.SessionKey,
			"channel":     target.Channel,
			"chat_id":     target.ChatID,
			"queue_depth": al.pendingSteeringCountForScope(target.SessionKey),
		})

	continued, err := al.drainQueuedSteeringContinuations(ctx, &target)
	if err != nil {
		logger.WarnCF("agent", "Failed to recover stranded steering messages",
			map[string]any{
				"session_key": target.SessionKey,
				"error":       err.Error(),
			})
		return
	}
	if continued != "" {
		al.PublishResponseIfNeeded(ctx, target.Channel, target.ChatID, target.SessionKey, continued)
	}
}

func (al *AgentLoop) processMessageSync(ctx context.Context, msg bus.InboundMessage) {
	if al.channelManager != nil {
		defer al.channelManager.InvokeTypingStop(msg.Channel, msg.ChatID)
	}

	response, err := al.processMessage(ctx, msg)
	al.publishResponseOrError(ctx, msg.Channel, msg.ChatID, msg.SessionKey, response, err)
}

func (al *AgentLoop) runTurnWithSteering(ctx context.Context, initialMsg bus.InboundMessage) {
	// Process the initial message
	response, err := al.processMessage(ctx, initialMsg)
	if err != nil {
		if !al.maybePublishError(ctx, initialMsg.Channel, initialMsg.ChatID, initialMsg.SessionKey, err) {
			return // context canceled
		}
		response = ""
	}
	finalResponse := response

	// Build continuation target
	target, targetErr := al.buildContinuationTarget(initialMsg)
	if targetErr != nil {
		logger.WarnCF("agent", "Failed to build steering continuation target",
			map[string]any{
				"channel": initialMsg.Channel,
				"error":   targetErr.Error(),
			})
		return
	}
	if target == nil {
		// System message or non-routable, response already published
		return
	}

	continued, continueErr := al.drainQueuedSteeringContinuations(ctx, target)
	if continueErr != nil {
		logger.WarnCF("agent", "Failed to continue queued steering",
			map[string]any{
				"channel": target.Channel,
				"chat_id": target.ChatID,
				"error":   continueErr.Error(),
			})
	} else if continued != "" {
		finalResponse = continued
	}

	// Publish final response
	if finalResponse != "" {
		al.PublishResponseIfNeeded(ctx, target.Channel, target.ChatID, target.SessionKey, finalResponse)
	}
}

func (al *AgentLoop) drainQueuedSteeringContinuations(
	ctx context.Context,
	target *continuationTarget,
) (string, error) {
	if target == nil {
		return "", nil
	}

	finalResponse := ""
	for al.pendingSteeringCountForScope(target.SessionKey) > 0 {
		if err := ctx.Err(); err != nil {
			return finalResponse, err
		}

		logger.InfoCF("agent", "Continuing queued steering after turn end",
			map[string]any{
				"channel":     target.Channel,
				"chat_id":     target.ChatID,
				"session_key": target.SessionKey,
				"queue_depth": al.pendingSteeringCountForScope(target.SessionKey),
			})

		continued, continueErr := al.Continue(ctx, target.SessionKey, target.Channel, target.ChatID)
		if continueErr != nil {
			return finalResponse, continueErr
		}
		if continued == "" {
			break
		}
		finalResponse = continued
	}

	return finalResponse, nil
}

func (al *AgentLoop) resolveSteeringTarget(msg bus.InboundMessage) (string, string, bool) {
	if msg.Channel == "system" {
		return "", "", false
	}

	route, agent, err := al.resolveMessageRoute(msg)
	if err != nil || agent == nil {
		return "", "", false
	}
	allocation := al.allocateRouteSession(route, msg)

	return resolveScopeKey(allocation.SessionKey, msg.SessionKey), agent.ID, true
}
