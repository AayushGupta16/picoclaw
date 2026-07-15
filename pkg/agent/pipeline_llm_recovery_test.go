package agent

import (
	"context"
	"testing"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
)

func TestIsRecoverableEmptyResponse(t *testing.T) {
	tests := []struct {
		name string
		resp *providers.LLMResponse
		want bool
	}{
		{
			name: "nil response",
			resp: nil,
			want: false,
		},
		{
			name: "empty content, no tools, end_turn is recoverable",
			resp: &providers.LLMResponse{Content: "", FinishReason: "stop"},
			want: true,
		},
		{
			name: "whitespace-only content is recoverable",
			resp: &providers.LLMResponse{Content: "  \n\t ", FinishReason: "stop"},
			want: true,
		},
		{
			name: "refusal is not retried",
			resp: &providers.LLMResponse{Content: "", FinishReason: "refusal"},
			want: false,
		},
		{
			name: "has text content",
			resp: &providers.LLMResponse{Content: "hello", FinishReason: "stop"},
			want: false,
		},
		{
			name: "has tool calls",
			resp: &providers.LLMResponse{
				Content:      "",
				FinishReason: "tool_calls",
				ToolCalls:    []providers.ToolCall{{Name: "search"}},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isRecoverableEmptyResponse(tt.resp); got != tt.want {
				t.Errorf("isRecoverableEmptyResponse() = %v, want %v", got, tt.want)
			}
		})
	}
}

// sequencedChatProvider returns one planned response per Chat call, in order.
// Unlike configuredStreamingProvider's by-model map, this lets a single model
// return different responses across retries.
type sequencedChatProvider struct {
	chatCalls  int
	chatModels []string
	plan       []*providers.LLMResponse
}

func (p *sequencedChatProvider) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	opts map[string]any,
) (*providers.LLMResponse, error) {
	p.chatCalls++
	p.chatModels = append(p.chatModels, model)
	if p.chatCalls <= len(p.plan) {
		return p.plan[p.chatCalls-1], nil
	}
	return &providers.LLMResponse{Content: "chat response"}, nil
}

func (p *sequencedChatProvider) GetDefaultModel() string {
	return "mock-model"
}

// newRefusalFailoverTestConfig builds the streaming test config (streaming off)
// with refusal failover pointed at a "failover-model" model_list entry.
func newRefusalFailoverTestConfig(t *testing.T, fallbacks []string) *config.Config {
	t.Helper()
	cfg := newConfiguredStreamingTestConfig(t, false, false, fallbacks)
	cfg.Agents.Defaults.RefusalFailover = &config.RefusalFailoverConfig{Model: "failover-model"}
	cfg.ModelList = append(cfg.ModelList, &config.ModelConfig{
		ModelName: "failover-model",
		Provider:  "openai",
		Model:     "openai/failover-model",
	})
	return cfg
}

func TestRefusalFailsOverToConfiguredModelAndArmsHold(t *testing.T) {
	cfg := newRefusalFailoverTestConfig(t, nil)
	msgBus := bus.NewMessageBus()
	provider := &configuredStreamingProvider{
		chatResponsesByModel: map[string]*providers.LLMResponse{
			"openai/test-model":     {Content: "", FinishReason: "refusal"},
			"openai/failover-model": {Content: "recovered by failover"},
		},
	}
	al := NewAgentLoop(cfg, msgBus, provider)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != "recovered by failover" {
		t.Fatalf("response = %q, want recovered by failover", got)
	}
	if len(provider.chatModels) != 2 {
		t.Fatalf("chat models = %v, want 2 calls (refusal fails over immediately)", provider.chatModels)
	}
	if provider.chatModels[1] != "openai/failover-model" {
		t.Fatalf("second call model = %q, want openai/failover-model", provider.chatModels[1])
	}
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); !active {
		t.Fatal("refusal hold should be armed after the primary refused")
	}
}

func TestRefusalFromFailoverModelSurfacesNoticeWithoutLooping(t *testing.T) {
	cfg := newRefusalFailoverTestConfig(t, nil)
	msgBus := bus.NewMessageBus()
	provider := &configuredStreamingProvider{
		chatResponsesByModel: map[string]*providers.LLMResponse{
			"openai/test-model":     {Content: "", FinishReason: "refusal"},
			"openai/failover-model": {Content: "", FinishReason: "refusal"},
		},
	}
	al := NewAgentLoop(cfg, msgBus, provider)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != refusalResponse {
		t.Fatalf("response = %q, want refusal notice %q", got, refusalResponse)
	}
	if len(provider.chatModels) != 2 {
		t.Fatalf("chat models = %v, want exactly 2 calls (no failover loop)", provider.chatModels)
	}
	if provider.chatModels[1] != "openai/failover-model" {
		t.Fatalf("second call model = %q, want openai/failover-model", provider.chatModels[1])
	}
}

func TestRefusalWithoutFailoverConfiguredSurfacesNotice(t *testing.T) {
	cfg := newConfiguredStreamingTestConfig(t, false, false, nil)
	msgBus := bus.NewMessageBus()
	provider := &configuredStreamingProvider{
		chatResponsesByModel: map[string]*providers.LLMResponse{
			"openai/test-model": {Content: "", FinishReason: "refusal"},
		},
	}
	al := NewAgentLoop(cfg, msgBus, provider)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != refusalResponse {
		t.Fatalf("response = %q, want refusal notice %q", got, refusalResponse)
	}
	if provider.chatCalls != 1 {
		t.Fatalf("Chat calls = %d, want 1 (refusal is not retried without a failover)", provider.chatCalls)
	}
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); active {
		t.Fatal("refusal hold should not arm without a configured failover")
	}
}

func TestRefusalWithRetryBudgetExhaustedStillArmsHold(t *testing.T) {
	cfg := newRefusalFailoverTestConfig(t, nil)
	cfg.Agents.Defaults.MaxLLMRetries = 1
	msgBus := bus.NewMessageBus()
	// The empty response burns the whole retry budget before the refusal
	// arrives, so no failover attempt is left for this turn — the hold must
	// still arm so the next turns skip the refusing primary.
	provider := &sequencedChatProvider{plan: []*providers.LLMResponse{
		{Content: ""},
		{Content: "", FinishReason: "refusal"},
	}}
	al := NewAgentLoop(cfg, msgBus, provider)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != refusalResponse {
		t.Fatalf("response = %q, want refusal notice %q", got, refusalResponse)
	}
	if provider.chatCalls != 2 {
		t.Fatalf("Chat calls = %d, want 2 (no failover attempt left)", provider.chatCalls)
	}
	if _, active := al.GetRegistry().GetDefaultAgent().RefusalHoldUntil(); !active {
		t.Fatal("refusal hold must arm even when the retry budget is exhausted")
	}
}

func TestRefusalHoldFrontsFailoverCandidatesUntilExpiry(t *testing.T) {
	cfg := newRefusalFailoverTestConfig(t, []string{"fallback-model"})
	msgBus := bus.NewMessageBus()
	al := NewAgentLoop(cfg, msgBus, &configuredStreamingProvider{})
	agent := al.GetRegistry().GetDefaultAgent()

	candidates, model, usedLight := al.selectCandidates(agent, "hello", nil)
	if usedLight {
		t.Fatal("usedLight = true, want false without routing configured")
	}
	if model != "openai/test-model" {
		t.Fatalf("model before hold = %q, want openai/test-model", model)
	}

	agent.ArmRefusalHold(time.Hour)
	candidates, model, _ = al.selectCandidates(agent, "hello", nil)
	if model != "openai/failover-model" {
		t.Fatalf("model under hold = %q, want openai/failover-model", model)
	}
	if len(candidates) != 2 ||
		candidates[0].Model != "openai/failover-model" ||
		candidates[1].Model != "openai/fallback-model" {
		t.Fatalf("hold candidates = %v, want failover fronted with the primary dropped", candidates)
	}

	agent.refusalHold.mu.Lock()
	agent.refusalHold.until = time.Now().Add(-time.Minute)
	agent.refusalHold.mu.Unlock()
	candidates, model, _ = al.selectCandidates(agent, "hello", nil)
	if model != "openai/test-model" {
		t.Fatalf("model after hold expiry = %q, want openai/test-model", model)
	}
	if len(candidates) != 2 || candidates[0].Model != "openai/test-model" {
		t.Fatalf("candidates after hold expiry = %v, want the primary fronted again", candidates)
	}
}

func TestArmedHoldStartsTurnOnFailoverModel(t *testing.T) {
	cfg := newRefusalFailoverTestConfig(t, nil)
	msgBus := bus.NewMessageBus()
	provider := &configuredStreamingProvider{
		chatResponsesByModel: map[string]*providers.LLMResponse{
			"openai/test-model":     {Content: "primary reply"},
			"openai/failover-model": {Content: "failover reply"},
		},
	}
	al := NewAgentLoop(cfg, msgBus, provider)
	al.GetRegistry().GetDefaultAgent().ArmRefusalHold(time.Hour)

	got := runConfiguredStreamingTurn(t, al, "pico")

	if got != "failover reply" {
		t.Fatalf("response = %q, want failover reply", got)
	}
	if len(provider.chatModels) != 1 || provider.chatModels[0] != "openai/failover-model" {
		t.Fatalf("chat models = %v, want a single call on openai/failover-model", provider.chatModels)
	}
}
