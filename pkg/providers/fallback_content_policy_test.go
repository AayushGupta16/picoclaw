package providers

import (
	"context"
	"errors"
	"testing"
)

// A provider that rejects a prompt on content policy must not strand the turn:
// the chain has to move to the next candidate, which may well accept it.
func TestFallbackChain_ContentPolicy400AdvancesToNextCandidate(t *testing.T) {
	realErr := errors.New(`API request failed:
  Status: 400
  Body:   {
  "error": {
    "message": "This content was flagged for possible cybersecurity risk. If this seems wrong, try rephrasing your prompt.",
    "type": "invalid_request_error"
  }
}`)

	fc := NewFallbackChain(NewCooldownTracker(), nil)
	candidates := []FallbackCandidate{
		{Provider: "openai-responses", Model: "gpt-5.6-sol"},
		{Provider: "anthropic-messages", Model: "claude-opus-4-8"},
	}

	var tried []string
	res, err := fc.Execute(context.Background(), candidates,
		func(ctx context.Context, provider, model string) (*LLMResponse, error) {
			tried = append(tried, model)
			if model == "gpt-5.6-sol" {
				return nil, realErr
			}
			return &LLMResponse{Content: "opus handled it"}, nil
		})
	if err != nil {
		t.Fatalf("chain aborted instead of advancing: %v (tried=%v)", err, tried)
	}
	if len(tried) != 2 || tried[1] != "claude-opus-4-8" {
		t.Fatalf("expected advance to opus, tried=%v", tried)
	}
	if res.Response.Content != "opus handled it" {
		t.Fatalf("unexpected response: %+v", res.Response)
	}
	t.Logf("PASS: tried %v, recovered on %s", tried, res.Model)
}

// Control for the above: a structurally malformed request must still abort on
// the first candidate — no other model will accept the same broken payload.
func TestFallbackChain_MalformedRequestAbortsImmediately(t *testing.T) {
	badReq := errors.New(`bad request (400): {"type":"error","error":{"type":"invalid_request_error","message":"messages.4.content.0: unexpected ` + "`tool_use`" + ` block"}}`)

	fc := NewFallbackChain(NewCooldownTracker(), nil)
	candidates := []FallbackCandidate{
		{Provider: "anthropic-messages", Model: "claude-opus-4-8"},
		{Provider: "anthropic-messages", Model: "claude-opus-4-8-or"},
	}

	var tried []string
	_, err := fc.Execute(context.Background(), candidates,
		func(ctx context.Context, provider, model string) (*LLMResponse, error) {
			tried = append(tried, model)
			return nil, badReq
		})
	if err == nil {
		t.Fatal("expected abort on malformed request")
	}
	if len(tried) != 1 {
		t.Fatalf("malformed request should abort after ONE attempt, tried=%v", tried)
	}
	t.Logf("PASS: aborted after %v", tried)
}
