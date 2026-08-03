// PicoClaw - Ultra-lightweight personal AI agent

package agent

import (
	"context"
	"sync"

	"github.com/sipeed/picoclaw/pkg/bus"
)

// turnLane is the admission priority of a turn.
type turnLane int

const (
	// laneHuman is for turns started by an inbound message from a person.
	laneHuman turnLane = iota
	// laneBackground is for turns the agent generated for itself: cron
	// firings and async tool / subagent results.
	laneBackground
)

// turnGate admits turns up to a fixed concurrency, always handing a freed slot
// to a human waiter before a background one.
//
// Self-generated work regenerates itself — every finishing subagent posts a
// result that starts another turn, which refills the subagent slot — so a
// first-come queue lets it starve inbound messages indefinitely. Overtaking
// background waiters without limit is the point: a human message then waits
// only for the turns already in flight, never for the queue behind them.
type turnGate struct {
	mu         sync.Mutex
	capacity   int
	held       int
	human      []chan struct{}
	background []chan struct{}
}

func newTurnGate(capacity int) *turnGate {
	if capacity < 1 {
		capacity = 1
	}
	return &turnGate{capacity: capacity}
}

// acquire blocks until a slot is free or ctx is done. Every successful acquire
// must be paired with exactly one release.
func (g *turnGate) acquire(ctx context.Context, lane turnLane) error {
	g.mu.Lock()
	if g.held < g.capacity && g.canJumpQueueLocked(lane) {
		g.held++
		g.mu.Unlock()
		return nil
	}
	ready := make(chan struct{})
	if lane == laneHuman {
		g.human = append(g.human, ready)
	} else {
		g.background = append(g.background, ready)
	}
	g.mu.Unlock()

	select {
	case <-ready:
		return nil
	case <-ctx.Done():
		if !g.cancelWaiter(lane, ready) {
			// The slot was handed over between the cancellation and the
			// removal; give it back rather than leaking it.
			g.release()
		}
		return ctx.Err()
	}
}

// canJumpQueueLocked reports whether an arriving turn may take a free slot
// ahead of the waiters already queued. Waiters in the same lane are served in
// arrival order; background arrivals also yield to every human waiter.
func (g *turnGate) canJumpQueueLocked(lane turnLane) bool {
	if len(g.human) > 0 {
		return false
	}
	return lane == laneHuman || len(g.background) == 0
}

func (g *turnGate) release() {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.held > 0 {
		g.held--
	}
	for g.held < g.capacity {
		var ready chan struct{}
		switch {
		case len(g.human) > 0:
			ready, g.human = g.human[0], g.human[1:]
		case len(g.background) > 0:
			ready, g.background = g.background[0], g.background[1:]
		default:
			return
		}
		g.held++
		close(ready)
	}
}

// cancelWaiter drops a waiter that gave up. It returns false when the slot was
// already handed to that waiter, in which case the caller now owns the slot.
func (g *turnGate) cancelWaiter(lane turnLane, ready chan struct{}) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	queue := &g.human
	if lane == laneBackground {
		queue = &g.background
	}
	for i, waiter := range *queue {
		if waiter == ready {
			*queue = append((*queue)[:i], (*queue)[i+1:]...)
			return true
		}
	}
	return false
}

// snapshot returns the slots in use and the waiters queued in each lane.
func (g *turnGate) snapshot() (held, humanWaiting, backgroundWaiting int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.held, len(g.human), len(g.background)
}

// backgroundTurnQueue holds self-generated inbound messages until the
// background runner can take them.
//
// These used to run inline on the receive loop, which meant a subagent fan-out
// stopped the agent from dequeuing the message bus at all: inbound human
// messages sat in the bus buffer with no turn, no steering entry and no event
// to show for them. The queue is unbounded because the alternative is dropping
// a subagent result, which loses the work it reports; depth is bounded in
// practice by the agent's concurrent subagent limit.
type backgroundTurnQueue struct {
	mu      sync.Mutex
	pending []bus.InboundMessage
	signal  chan struct{}
}

func newBackgroundTurnQueue() *backgroundTurnQueue {
	return &backgroundTurnQueue{signal: make(chan struct{}, 1)}
}

func (q *backgroundTurnQueue) push(msg bus.InboundMessage) {
	q.mu.Lock()
	q.pending = append(q.pending, msg)
	q.mu.Unlock()

	select {
	case q.signal <- struct{}{}:
	default:
	}
}

func (q *backgroundTurnQueue) pop() (bus.InboundMessage, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.pending) == 0 {
		return bus.InboundMessage{}, false
	}
	msg := q.pending[0]
	q.pending[0] = bus.InboundMessage{}
	q.pending = q.pending[1:]
	return msg, true
}

func (q *backgroundTurnQueue) len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.pending)
}
