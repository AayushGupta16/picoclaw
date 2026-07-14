package agent

import (
	"testing"

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
