// PicoClaw - Ultra-lightweight personal AI agent
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package anthropicmessages

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers/common"
	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

type (
	ToolCall               = protocoltypes.ToolCall
	FunctionCall           = protocoltypes.FunctionCall
	LLMResponse            = protocoltypes.LLMResponse
	UsageInfo              = protocoltypes.UsageInfo
	Message                = protocoltypes.Message
	ContentBlock           = protocoltypes.ContentBlock
	CacheControl           = protocoltypes.CacheControl
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
)

const (
	defaultAPIVersion     = "2023-06-01"
	defaultBaseURL        = "https://api.anthropic.com/v1"
	defaultRequestTimeout = 120 * time.Second

	// maxCacheBreakpoints is Anthropic's limit on cache_control markers per
	// request (system blocks + message blocks combined).
	maxCacheBreakpoints = 4

	// breakpointStrideBlocks is the minimum number of content blocks between
	// consecutive message cache markers. Anthropic's cache lookup walks back
	// at most ~20 content blocks from a breakpoint to find a prior entry;
	// markers spaced closer than that chain — each one connects to the
	// previous marker's entry (or, within a single request, to the entry the
	// previous marker just created), so a cold cache can rebuild the whole
	// prefix instead of stranding everything past the system blocks. Spacing
	// them at 15 leaves headroom for a multi-block message straddling a gap.
	breakpointStrideBlocks = 15
)

// Provider implements Anthropic Messages API via HTTP (without SDK).
// It supports custom endpoints that use Anthropic's native message format.
type Provider struct {
	apiKey     string
	apiBase    string
	httpClient *http.Client
	userAgent  string
}

// NewProvider creates a new Anthropic Messages API provider.
func NewProvider(apiKey, apiBase, userAgent string) *Provider {
	return NewProviderWithTimeout(apiKey, apiBase, userAgent, 0)
}

// NewProviderWithTimeout creates a provider with custom request timeout.
func NewProviderWithTimeout(apiKey, apiBase, userAgent string, timeoutSeconds int) *Provider {
	baseURL := common.NormalizeBaseURL(apiBase, defaultBaseURL, true)
	timeout := defaultRequestTimeout
	if timeoutSeconds > 0 {
		timeout = time.Duration(timeoutSeconds) * time.Second
	}

	return &Provider{
		apiKey:    apiKey,
		apiBase:   baseURL,
		userAgent: userAgent,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// Chat sends messages to the Anthropic Messages API and returns the response.
func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("API key not configured")
	}

	// Build request body
	requestBody, err := buildRequestBody(messages, tools, model, options)
	if err != nil {
		return nil, fmt.Errorf("building request body: %w", err)
	}

	// Serialize to JSON
	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("serializing request body: %w", err)
	}

	// Build request URL
	endpointURL, err := url.JoinPath(p.apiBase, "messages")
	if err != nil {
		return nil, fmt.Errorf("building endpoint URL: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", endpointURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", p.apiKey) //nolint:canonicalheader // Anthropic API requires exact header name
	req.Header.Set("Anthropic-Version", defaultAPIVersion)
	if p.userAgent != "" {
		req.Header.Set("User-Agent", p.userAgent)
	}

	// Execute request
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	// Check for HTTP errors with detailed messages
	switch resp.StatusCode {
	case http.StatusUnauthorized:
		return nil, fmt.Errorf("authentication failed (401): check your API key")
	case http.StatusTooManyRequests:
		return nil, fmt.Errorf("rate limited (429): %s", string(body))
	case http.StatusBadRequest:
		return nil, fmt.Errorf("bad request (400): %s", string(body))
	case http.StatusNotFound:
		return nil, fmt.Errorf("endpoint not found (404): %s", string(body))
	case http.StatusInternalServerError:
		return nil, fmt.Errorf("internal server error (500): %s", string(body))
	case http.StatusServiceUnavailable:
		return nil, fmt.Errorf("service unavailable (503): %s", string(body))
	default:
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		}
	}

	// Parse response
	return parseResponseBody(body)
}

// GetDefaultModel returns the default model for this provider.
func (p *Provider) GetDefaultModel() string {
	return "claude-sonnet-4.6"
}

// buildRequestBody converts internal message format to Anthropic Messages API format.
func buildRequestBody(
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (map[string]any, error) {
	// max_tokens is required and guaranteed by agent loop
	maxTokens, ok := common.AsInt(options["max_tokens"])
	if !ok {
		return nil, fmt.Errorf("max_tokens is required in options")
	}

	result := map[string]any{
		"model":      model,
		"max_tokens": int64(maxTokens),
		"messages":   []any{},
	}

	// Set temperature from options. Anthropic's native Messages API rejects
	// temperature/top_p/top_k on Opus 4.7+ and Fable/Mythos 5 with a 400, so
	// sampling params are omitted entirely for those models.
	if modelRejectsSamplingParams(model) {
		if dropped := samplingParamsInOptions(options); len(dropped) > 0 {
			logger.DebugCF("provider.anthropic_messages",
				"dropping sampling params rejected by model", map[string]any{
					"model":  model,
					"params": dropped,
				})
		}
	} else if temp, ok := common.AsFloat(options["temperature"]); ok {
		result["temperature"] = temp
	}

	// Process messages
	var systemPrompt string
	var systemBlocks []any
	hasSystemParts := false
	var apiMessages []any

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// Accumulate system messages. Two representations are built in
			// parallel: a flat string (legacy behavior) and structured blocks.
			// When any system message carries SystemParts, the block form is
			// sent so per-block cache_control survives (fixes #2191); otherwise
			// the flat string preserves the existing concat semantics.
			if len(msg.SystemParts) > 0 {
				hasSystemParts = true
				for _, part := range msg.SystemParts {
					if part.Text == "" {
						continue
					}
					block := map[string]any{
						"type": "text",
						"text": part.Text,
					}
					if part.CacheControl != nil && part.CacheControl.Type == "ephemeral" {
						block["cache_control"] = map[string]any{"type": "ephemeral"}
					}
					systemBlocks = append(systemBlocks, block)
				}
			} else if msg.Content != "" {
				systemBlocks = append(systemBlocks, map[string]any{
					"type": "text",
					"text": msg.Content,
				})
			}
			if msg.Content != "" {
				if systemPrompt != "" {
					systemPrompt += "\n\n" + msg.Content
				} else {
					systemPrompt = msg.Content
				}
			}

		case "user":
			if msg.ToolCallID != "" {
				// Tool result message — merge into previous user message if it contains tool_results
				toolResultBlock := map[string]any{
					"type":        "tool_result",
					"tool_use_id": msg.ToolCallID,
					"content":     msg.Content,
				}
				if len(apiMessages) > 0 {
					if prev, ok := apiMessages[len(apiMessages)-1].(map[string]any); ok && prev["role"] == "user" {
						if content, ok := prev["content"].([]map[string]any); ok {
							prev["content"] = append(content, toolResultBlock)
							continue
						}
					}
				}
				apiMessages = append(apiMessages, map[string]any{
					"role":    "user",
					"content": []map[string]any{toolResultBlock},
				})
			} else {
				// Regular user message. Inline images in msg.Media become an
				// Anthropic content-block array; without media (or when nothing
				// in it parses) the plain-string form is preserved unchanged.
				var content any = msg.Content
				if len(msg.Media) > 0 {
					if blocks := buildUserContentBlocks(msg.Content, msg.Media); len(blocks) > 0 {
						content = blocks
					}
				}
				apiMessages = append(apiMessages, map[string]any{
					"role":    "user",
					"content": content,
				})
			}

		case "assistant":
			content := []any{}

			// Add text content if present
			if msg.Content != "" {
				content = append(content, map[string]any{
					"type": "text",
					"text": msg.Content,
				})
			}

			// Add tool_use blocks
			for _, tc := range msg.ToolCalls {
				// Resolve tool name: prefer tc.Name, fallback to tc.Function.Name
				// (tc.Name/tc.Arguments are json:"-" and may be empty when
				// history is reloaded from the session store)
				toolName := tc.Name
				if toolName == "" && tc.Function != nil {
					toolName = tc.Function.Name
				}
				if strings.TrimSpace(toolName) == "" {
					continue
				}

				// Resolve arguments: prefer tc.Arguments, fallback to parsing
				// tc.Function.Arguments
				input := tc.Arguments
				if input == nil && tc.Function != nil && tc.Function.Arguments != "" {
					if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
						input = map[string]any{}
					}
				}
				// Handle nil Arguments (GLM-4 may return null input)
				if input == nil {
					input = map[string]any{}
				}

				toolUse := map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  toolName,
					"input": input,
				}
				content = append(content, toolUse)
			}

			apiMessages = append(apiMessages, map[string]any{
				"role":    "assistant",
				"content": content,
			})

		case "tool":
			// Tool result (alternative format) — merge into previous user message if it contains tool_results
			toolResultBlock := map[string]any{
				"type":        "tool_result",
				"tool_use_id": msg.ToolCallID,
				"content":     msg.Content,
			}
			if len(apiMessages) > 0 {
				if prev, ok := apiMessages[len(apiMessages)-1].(map[string]any); ok && prev["role"] == "user" {
					if content, ok := prev["content"].([]map[string]any); ok {
						prev["content"] = append(content, toolResultBlock)
						continue
					}
				}
			}
			apiMessages = append(apiMessages, map[string]any{
				"role":    "user",
				"content": []map[string]any{toolResultBlock},
			})
		}
	}

	result["messages"] = apiMessages

	// Set system prompt if present. Structured blocks are only sent when a
	// system message actually provided SystemParts; the flat string keeps
	// byte-identical behavior for callers that never populate them.
	if hasSystemParts && len(systemBlocks) > 0 {
		result["system"] = systemBlocks

		// Rolling conversation breakpoints: mark the most recent user turns so
		// the growing history prefix is cached incrementally across requests.
		// SystemParts presence is the cache-aware caller's opt-in signal, so
		// message markers are scoped to it as well.
		systemBreakpoints := clampSystemCacheBreakpoints(systemBlocks, maxCacheBreakpoints)
		applyMessageCacheBreakpoints(apiMessages, maxCacheBreakpoints-systemBreakpoints)
	} else if systemPrompt != "" {
		result["system"] = systemPrompt
	}

	// Add tools if present
	if len(tools) > 0 {
		result["tools"] = buildTools(tools)
	}

	return result, nil
}

// maxImageBytes is Anthropic's per-image decoded size limit (~5MB).
const maxImageBytes = 5 * 1024 * 1024

// buildUserContentBlocks converts a user message with inline media into
// Anthropic content blocks: an optional leading text block followed by one
// image block per parseable data:image/... URL. Entries that are not base64
// image data URLs of a supported subtype (jpeg/png/gif/webp), or whose decoded
// payload would exceed Anthropic's ~5MB image limit, are skipped with a
// warning. Returns nil when no blocks result, so the caller can fall back to
// plain-string content. Parsing mirrors bedrock's buildUserContent, but the
// payload stays base64-encoded — the API accepts the string directly.
func buildUserContentBlocks(text string, media []string) []map[string]any {
	var blocks []map[string]any
	if text != "" {
		blocks = append(blocks, map[string]any{
			"type": "text",
			"text": text,
		})
	}

	for _, mediaURL := range media {
		if !strings.HasPrefix(mediaURL, "data:image/") {
			logger.WarnCF("provider.anthropic_messages",
				"skipping media entry: not an image data URL", map[string]any{
					"prefix": truncateForLog(mediaURL),
				})
			continue
		}

		// Parse data URL: data:image/png;base64,<data>
		parts := strings.SplitN(mediaURL, ",", 2)
		if len(parts) != 2 || !strings.Contains(parts[0], ";base64") {
			logger.WarnCF("provider.anthropic_messages",
				"skipping media entry: image data URL is not base64-encoded", map[string]any{
					"header": truncateForLog(parts[0]),
				})
			continue
		}

		subtype := strings.TrimPrefix(parts[0], "data:image/")
		if idx := strings.Index(subtype, ";"); idx != -1 {
			subtype = subtype[:idx]
		}
		switch subtype {
		case "jpg":
			subtype = "jpeg"
		case "jpeg", "png", "gif", "webp":
		default:
			logger.WarnCF("provider.anthropic_messages",
				"skipping media entry: unsupported image type", map[string]any{
					"subtype": subtype,
				})
			continue
		}

		if decodedLen := base64.StdEncoding.DecodedLen(len(parts[1])); decodedLen > maxImageBytes {
			logger.WarnCF("provider.anthropic_messages",
				"skipping media entry: image exceeds size limit", map[string]any{
					"decoded_bytes": decodedLen,
					"limit_bytes":   maxImageBytes,
				})
			continue
		}

		blocks = append(blocks, map[string]any{
			"type": "image",
			"source": map[string]any{
				"type":       "base64",
				"media_type": "image/" + subtype,
				"data":       parts[1],
			},
		})
	}

	return blocks
}

// truncateForLog caps a media string for log output so a full base64 payload
// never lands in the logs.
func truncateForLog(s string) string {
	const maxLen = 48
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

// samplingRestrictedModelFamilies lists the model families whose native
// Messages API rejects temperature/top_p/top_k with a 400 error.
var samplingRestrictedModelFamilies = []string{
	"opus-4-7",
	"opus-4-8",
	"fable-5",
	"mythos-5",
}

// modelRejectsSamplingParams reports whether the given model rejects sampling
// parameters. Matching is case-insensitive and tolerates dotted config-style
// IDs (e.g. "claude-opus-4.8" as well as "claude-opus-4-8").
func modelRejectsSamplingParams(model string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(model, ".", "-"))
	for _, family := range samplingRestrictedModelFamilies {
		if strings.Contains(normalized, family) {
			return true
		}
	}
	return false
}

// samplingParamsInOptions returns which sampling-parameter keys are present in
// the request options, for drop logging.
func samplingParamsInOptions(options map[string]any) []string {
	var present []string
	for _, key := range []string{"temperature", "top_p", "top_k"} {
		if _, ok := options[key]; ok {
			present = append(present, key)
		}
	}
	return present
}

// clampSystemCacheBreakpoints enforces Anthropic's cache_control marker limit
// on the system block array, stripping markers from the EARLIEST blocks when
// the caller marked more than the limit. The last markers are kept because a
// breakpoint caches the entire prefix before it — the final marked block
// covers everything an earlier marker would have, so dropping early markers
// only removes intermediate read points, never coverage. Returns the number of
// markers retained.
func clampSystemCacheBreakpoints(systemBlocks []any, limit int) int {
	marked := make([]map[string]any, 0, len(systemBlocks))
	for _, b := range systemBlocks {
		if block, ok := b.(map[string]any); ok {
			if _, hasMarker := block["cache_control"]; hasMarker {
				marked = append(marked, block)
			}
		}
	}
	if len(marked) <= limit {
		return len(marked)
	}
	for _, block := range marked[:len(marked)-limit] {
		delete(block, "cache_control")
	}
	return limit
}

// applyMessageCacheBreakpoints attaches {"cache_control": {"type": "ephemeral"}}
// to the last content block of user-role messages (including tool_result user
// messages), walking backwards from the newest message until the breakpoint
// budget is exhausted. The newest user message is always marked — that is what
// extends the cached prefix as the conversation grows. Earlier markers are
// placed only after breakpointStrideBlocks content blocks have accumulated
// since the previous marker, so consecutive breakpoints stay within
// Anthropic's ~20-block cache lookback of each other: on a cold cache the
// chain rebuilds front-to-back in one request, and on a warm cache each new
// tail marker connects to the previous request's tail. (The prior behavior —
// marking the N most recent user messages with no spacing — clustered every
// marker at the tail; after any cache miss the gap back to the system blocks
// exceeded the lookback and conversation content never got cached at all.)
// String content is converted to a single-element text block array so it can
// carry the marker.
func applyMessageCacheBreakpoints(apiMessages []any, budget int) {
	blocksSinceMarker := 0
	marked := 0

	mark := func(msg map[string]any) bool {
		switch content := msg["content"].(type) {
		case string:
			if content == "" {
				return false
			}
			msg["content"] = []map[string]any{{
				"type":          "text",
				"text":          content,
				"cache_control": map[string]any{"type": "ephemeral"},
			}}
			return true
		case []map[string]any:
			if len(content) == 0 {
				return false
			}
			content[len(content)-1]["cache_control"] = map[string]any{"type": "ephemeral"}
			return true
		}
		return false
	}

	for i := len(apiMessages) - 1; i >= 0 && marked < budget; i-- {
		msg, ok := apiMessages[i].(map[string]any)
		if !ok {
			continue
		}

		if msg["role"] == "user" &&
			(marked == 0 || blocksSinceMarker >= breakpointStrideBlocks) &&
			mark(msg) {
			marked++
			// The marker sits on this message's LAST block; its earlier
			// blocks lie between this marker and the next-older one, so they
			// count toward the next gap.
			blocksSinceMarker = messageBlockCount(msg) - 1
			continue
		}

		blocksSinceMarker += messageBlockCount(msg)
	}
}

// messageBlockCount reports how many content blocks a message contributes to
// the request, for breakpoint-stride accounting. String content is one text
// block; block arrays count their elements.
func messageBlockCount(msg map[string]any) int {
	switch content := msg["content"].(type) {
	case string:
		if content == "" {
			return 0
		}
		return 1
	case []map[string]any:
		return len(content)
	case []any:
		return len(content)
	}
	return 0
}

// buildTools converts tool definitions to Anthropic format.
func buildTools(tools []ToolDefinition) []any {
	result := make([]any, len(tools))
	for i, tool := range tools {
		toolDef := map[string]any{
			"name":         tool.Function.Name,
			"description":  tool.Function.Description,
			"input_schema": tool.Function.Parameters,
		}
		result[i] = toolDef
	}
	return result
}

// parseResponseBody parses Anthropic Messages API response.
func parseResponseBody(body []byte) (*LLMResponse, error) {
	var resp anthropicMessageResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing JSON response: %w", err)
	}

	// Extract content and tool calls
	var content strings.Builder
	toolCalls := make([]ToolCall, 0) // Initialize as empty slice (not nil) for consistent JSON serialization

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "tool_use":
			argsJSON, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
				Function: &FunctionCall{
					Name:      block.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Map stop_reason
	finishReason := "stop"
	switch resp.StopReason {
	case "tool_use":
		finishReason = "tool_calls"
	case "max_tokens":
		finishReason = "length"
	case "end_turn":
		finishReason = "stop"
	case "stop_sequence":
		finishReason = "stop"
	}

	return &LLMResponse{
		Content:      content.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage: &UsageInfo{
			PromptTokens:             int(resp.Usage.InputTokens),
			CompletionTokens:         int(resp.Usage.OutputTokens),
			TotalTokens:              int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
			CacheReadInputTokens:     int(resp.Usage.CacheReadInputTokens),
			CacheCreationInputTokens: int(resp.Usage.CacheCreationInputTokens),
		},
	}, nil
}

// Anthropic API response structures

type anthropicMessageResponse struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Model      string         `json:"model"`
	Usage      usageInfo      `json:"usage"`
}

type contentBlock struct {
	Type  string         `json:"type"`
	Text  string         `json:"text,omitempty"`
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
}

type usageInfo struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}
