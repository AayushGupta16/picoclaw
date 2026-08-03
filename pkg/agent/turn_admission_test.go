package agent

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	runtimeevents "github.com/sipeed/picoclaw/pkg/events"
	"github.com/sipeed/picoclaw/pkg/providers"
)

// turnOrderProvider records the last user message of every turn it is asked to
// answer and can hold the first one open, so a test can pin one turn in flight
// while it queues more behind it.
type turnOrderProvider struct {
	mu           sync.Mutex
	calls        int
	seen         []string
	firstStarted chan struct{}
	releaseFirst chan struct{}
	startedOnce  sync.Once
	releaseOnce  sync.Once
}

func newTurnOrderProvider() *turnOrderProvider {
	return &turnOrderProvider{
		firstStarted: make(chan struct{}),
		releaseFirst: make(chan struct{}),
	}
}

func (p *turnOrderProvider) release() {
	p.releaseOnce.Do(func() { close(p.releaseFirst) })
}

func (p *turnOrderProvider) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	opts map[string]any,
) (*providers.LLMResponse, error) {
	p.mu.Lock()
	p.calls++
	call := p.calls
	p.seen = append(p.seen, lastUserContent(messages))
	p.mu.Unlock()

	if call == 1 {
		p.startedOnce.Do(func() { close(p.firstStarted) })
		<-p.releaseFirst
	}
	return &providers.LLMResponse{Content: "ok"}, nil
}

func (p *turnOrderProvider) GetDefaultModel() string { return "turn-order-mock" }

func (p *turnOrderProvider) snapshot() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.seen...)
}

func lastUserContent(messages []providers.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return messages[i].Content
		}
	}
	return ""
}

func newAdmissionTestLoop(t *testing.T, provider providers.LLMProvider) (*AgentLoop, *bus.MessageBus) {
	t.Helper()
	// Not t.TempDir(): these tests leave turns running when they return, and
	// its cleanup fails the test if the workspace is still being written to.
	workspace, err := os.MkdirTemp("", "turn-admission-*")
	if err != nil {
		t.Fatalf("create workspace: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(workspace) })

	cfg := &config.Config{
		Agents: config.AgentsConfig{
			Defaults: config.AgentDefaults{
				Workspace:         workspace,
				ModelName:         "test-model",
				MaxTokens:         4096,
				MaxToolIterations: 10,
			},
		},
	}
	msgBus := bus.NewMessageBus()
	return NewAgentLoop(cfg, msgBus, provider), msgBus
}

func waitForGate(t *testing.T, al *AgentLoop, wantHuman, wantBackground int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		_, human, background := al.turns.snapshot()
		if human >= wantHuman && background >= wantBackground {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	_, human, background := al.turns.snapshot()
	t.Fatalf(
		"gate never reached the expected waiters: human=%d (want %d), background=%d (want %d)",
		human, wantHuman, background, wantBackground,
	)
}

func TestTurnGate_HumanWaiterWinsSaturatedBackgroundQueue(t *testing.T) {
	gate := newTurnGate(1)
	ctx := context.Background()

	if err := gate.acquire(ctx, laneBackground); err != nil {
		t.Fatalf("acquire in-flight turn: %v", err)
	}

	admitted := make(chan string, 9)
	for i := 0; i < 8; i++ {
		go func() {
			if err := gate.acquire(ctx, laneBackground); err != nil {
				return
			}
			admitted <- "background"
		}()
	}

	// Queue the human waiter only once every background turn is already in
	// line, so winning cannot be an artifact of arrival order.
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, _, background := gate.snapshot(); background == 8 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("background waiters never queued")
		}
		time.Sleep(time.Millisecond)
	}

	go func() {
		if err := gate.acquire(ctx, laneHuman); err != nil {
			return
		}
		admitted <- "human"
	}()
	for {
		if _, human, _ := gate.snapshot(); human == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("human waiter never queued")
		}
		time.Sleep(time.Millisecond)
	}

	gate.release()

	select {
	case got := <-admitted:
		if got != "human" {
			t.Fatalf("first admitted after release = %q, want human", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("nothing was admitted after release")
	}
}

func TestTurnGate_ServesEachLaneInArrivalOrder(t *testing.T) {
	gate := newTurnGate(1)
	ctx := context.Background()

	if err := gate.acquire(ctx, laneHuman); err != nil {
		t.Fatalf("acquire in-flight turn: %v", err)
	}

	order := make(chan int, 3)
	for i := 1; i <= 3; i++ {
		go func(rank int) {
			if err := gate.acquire(ctx, laneHuman); err != nil {
				return
			}
			order <- rank
			gate.release()
		}(i)
		deadline := time.Now().Add(5 * time.Second)
		for {
			if _, human, _ := gate.snapshot(); human == i {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("waiter %d never queued", i)
			}
			time.Sleep(time.Millisecond)
		}
	}

	gate.release()
	for want := 1; want <= 3; want++ {
		select {
		case got := <-order:
			if got != want {
				t.Fatalf("admitted %d, want %d", got, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("waiter %d never admitted", want)
		}
	}
}

func TestTurnGate_CanceledWaiterDoesNotLeakSlot(t *testing.T) {
	gate := newTurnGate(1)

	if err := gate.acquire(context.Background(), laneHuman); err != nil {
		t.Fatalf("acquire in-flight turn: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	waited := make(chan error, 1)
	go func() { waited <- gate.acquire(ctx, laneBackground) }()

	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, _, background := gate.snapshot(); background == 1 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("waiter never queued")
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	if err := <-waited; err == nil {
		t.Fatal("canceled acquire returned nil error")
	}

	gate.release()
	if held, _, _ := gate.snapshot(); held != 0 {
		t.Fatalf("held = %d after release, want 0", held)
	}
	if err := gate.acquire(context.Background(), laneHuman); err != nil {
		t.Fatalf("gate did not admit after a canceled waiter: %v", err)
	}
}

// TestAgentLoop_Run_KeepsDrainingBusDuringSelfGeneratedTurn pins the first
// half of the failure: while a subagent result is being answered, an inbound
// message still has to be taken off the bus. Running that answer on the
// receive loop left founder messages sitting in the bus buffer with no turn,
// no steering entry and no event anywhere.
func TestAgentLoop_Run_KeepsDrainingBusDuringSelfGeneratedTurn(t *testing.T) {
	provider := newTurnOrderProvider()
	defer provider.release()
	al, msgBus := newAdmissionTestLoop(t, provider)

	runCtx, cancelRun := context.WithCancel(context.Background())
	defer cancelRun()
	go func() { _ = al.Run(runCtx) }()

	pubCtx, pubCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer pubCancel()

	if err := msgBus.PublishInbound(pubCtx, bus.InboundMessage{
		Context: bus.InboundContext{
			Channel:  "system",
			ChatID:   "test:chat1",
			ChatType: "direct",
			SenderID: "async:spawn",
		},
		Content: "spawn-result-0",
	}); err != nil {
		t.Fatalf("publish spawn result: %v", err)
	}
	select {
	case <-provider.firstStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("self-generated turn never started")
	}

	if err := msgBus.PublishInbound(pubCtx, bus.InboundMessage{
		Context: bus.InboundContext{
			Channel:  "test",
			ChatID:   "chat1",
			ChatType: "direct",
			SenderID: "founder",
		},
		Content: "STOP THE HUNTING",
	}); err != nil {
		t.Fatalf("publish founder message: %v", err)
	}

	waitForGate(t, al, 1, 0)
	provider.release()
}

// TestAgentLoop_Run_FounderMessageWinsSelfGeneratedBacklog reproduces the
// control failure: a subagent fan-out that refills its own slots plus a
// recurring cron, with founder messages arriving into the middle of it.
func TestAgentLoop_Run_FounderMessageWinsSelfGeneratedBacklog(t *testing.T) {
	provider := newTurnOrderProvider()
	defer provider.release()
	al, msgBus := newAdmissionTestLoop(t, provider)

	runCtx, cancelRun := context.WithCancel(context.Background())
	defer cancelRun()
	go func() { _ = al.Run(runCtx) }()

	pubCtx, pubCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer pubCancel()

	spawnResult := func(label string) bus.InboundMessage {
		return bus.InboundMessage{
			Context: bus.InboundContext{
				Channel:  "system",
				ChatID:   "test:chat1",
				ChatType: "direct",
				SenderID: "async:spawn",
			},
			Content: label,
		}
	}

	// One subagent result is in flight and holds the only turn slot.
	if err := msgBus.PublishInbound(pubCtx, spawnResult("spawn-result-0")); err != nil {
		t.Fatalf("publish first spawn result: %v", err)
	}
	select {
	case <-provider.firstStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("first self-generated turn never started")
	}

	// Its siblings finish behind it, exactly as a refilled fan-out does.
	for i := 1; i <= 4; i++ {
		if err := msgBus.PublishInbound(pubCtx, spawnResult(fmt.Sprintf("spawn-result-%d", i))); err != nil {
			t.Fatalf("publish spawn result %d: %v", i, err)
		}
	}

	// And the recurring cron fires through its real entry point.
	for i := 1; i <= 2; i++ {
		go func(job int) {
			_, _ = al.ProcessDirectWithChannel(
				runCtx,
				fmt.Sprintf("cron-job-%d", job),
				fmt.Sprintf("agent:cron-%d", job),
				"test",
				"chat-cron",
			)
		}(i)
	}
	waitForGate(t, al, 0, 2)

	// The founder message lands last, into the fully saturated queue.
	founder := bus.InboundMessage{
		Context: bus.InboundContext{
			Channel:  "test",
			ChatID:   "chat1",
			ChatType: "direct",
			SenderID: "founder",
		},
		Content: "STOP THE HUNTING",
	}
	if err := msgBus.PublishInbound(pubCtx, founder); err != nil {
		t.Fatalf("publish founder message: %v", err)
	}
	// Reaching the gate at all is the first half of the fix: the receive loop
	// has to still be draining the bus while a self-generated turn runs.
	waitForGate(t, al, 1, 2)

	provider.release()

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		seen := provider.snapshot()
		if len(seen) < 2 {
			time.Sleep(5 * time.Millisecond)
			continue
		}
		if !strings.Contains(seen[1], "STOP THE HUNTING") {
			t.Fatalf("turn after the in-flight one answered %q, want the founder message", seen[1])
		}
		return
	}
	t.Fatalf("no turn ran after the in-flight one; provider saw %v", provider.snapshot())
}

// TestAgentLoop_Run_RecoversStrandedFounderMessage covers the second half: a
// message queued for a session whose turn ended without draining it (the scopes
// a fan-out actually runs under — the main session and cron — never poll the
// founder's scope) must still get a turn.
func TestAgentLoop_Run_RecoversStrandedFounderMessage(t *testing.T) {
	provider := newTurnOrderProvider()
	provider.release() // nothing needs to be held open here
	al, _ := newAdmissionTestLoop(t, provider)

	runCtx, cancelRun := context.WithCancel(context.Background())
	defer cancelRun()
	go func() { _ = al.Run(runCtx) }()

	sessionKey, agentID, ok := al.resolveSteeringTarget(bus.InboundMessage{
		Context: bus.InboundContext{
			Channel:  "test",
			ChatID:   "chat1",
			ChatType: "direct",
			SenderID: "founder",
		},
	})
	if !ok {
		t.Fatal("expected the founder message to resolve to a session")
	}

	al.rememberSteeringTarget(continuationTarget{
		SessionKey: sessionKey,
		Channel:    "test",
		ChatID:     "chat1",
	})
	if err := al.enqueueSteeringMessage(sessionKey, agentID, providers.Message{
		Role:    "user",
		Content: "stranded founder message",
	}); err != nil {
		t.Fatalf("enqueue steering: %v", err)
	}

	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		for _, seen := range provider.snapshot() {
			if strings.Contains(seen, "stranded founder message") {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("stranded message never reached a turn; provider saw %v", provider.snapshot())
}

func TestEnqueueSteeringMessage_FullQueueEmitsError(t *testing.T) {
	al, _ := newAdmissionTestLoop(t, newTurnOrderProvider())

	events := make(chan string, 8)
	sub, err := al.RuntimeEventBus().Channel().Subscribe(
		context.Background(),
		runtimeevents.SubscribeOptions{
			Name:         "full-queue-watch",
			Buffer:       MaxQueueSize + 4,
			Concurrency:  runtimeevents.Locked,
			Backpressure: runtimeevents.Block,
		},
		func(_ context.Context, evt runtimeevents.Event) error {
			if evt.Kind == runtimeevents.KindAgentError {
				select {
				case events <- evt.Correlation.TraceID:
				default:
				}
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("subscribe: %v", err)
	}
	defer sub.Close()

	const scope = "sk_v1_full"
	for i := 0; i < MaxQueueSize; i++ {
		if err := al.enqueueSteeringMessage(scope, "main", providers.Message{
			Role:    "user",
			Content: "filler",
		}); err != nil {
			t.Fatalf("enqueue %d: %v", i, err)
		}
	}

	if err := al.enqueueSteeringMessage(scope, "main", providers.Message{
		Role:    "user",
		Content: "dropped founder message",
	}); err == nil {
		t.Fatal("expected the overflowing message to be rejected")
	}

	select {
	case tracePath := <-events:
		if tracePath != "turn.interrupt.dropped" {
			t.Fatalf("trace path = %q, want turn.interrupt.dropped", tracePath)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a dropped human message emitted no event")
	}
}
