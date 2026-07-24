package fstools

import (
	"context"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/media"
	_ "golang.org/x/image/bmp"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// Anthropic rejects any side over 2,000 px once a request contains enough
// images to use its many-image limits. Keep a little headroom so provider-side
// metadata or rounding can never turn a valid load_image result into a 400.
const maxVisionImageDimension = 1900

// LoadImageTool loads a local image file into the MediaStore and returns a
// media:// reference. The agent loop's resolveMediaRefs will then base64-encode
// it and attach it as an image_url part in the next LLM request, enabling
// vision on local files — the same pipeline used when a user sends an image
// through a chat channel.
//
// This is intentionally different from SendFileTool:
//   - SendFileTool  → MediaResult + WithResponseHandled() → sends file to user, ends turn
//   - LoadImageTool → plain ToolResult with media:// in ForLLM  → LLM sees the image next turn
type LoadImageTool struct {
	workspace   string
	restrict    bool
	maxFileSize int
	mediaStore  media.MediaStore
	allowPaths  []*regexp.Regexp

	defaultChannel string
	defaultChatID  string
}

func NewLoadImageTool(
	workspace string,
	restrict bool,
	maxFileSize int,
	store media.MediaStore,
	allowPaths ...[]*regexp.Regexp,
) *LoadImageTool {
	if maxFileSize <= 0 {
		maxFileSize = config.DefaultMaxMediaSize
	}
	var patterns []*regexp.Regexp
	if len(allowPaths) > 0 {
		patterns = allowPaths[0]
	}
	return &LoadImageTool{
		workspace:   workspace,
		restrict:    restrict,
		maxFileSize: maxFileSize,
		mediaStore:  store,
		allowPaths:  patterns,
	}
}

func (t *LoadImageTool) Name() string { return "load_image" }

func (t *LoadImageTool) Description() string {
	return "Load a local image file so you can analyze its contents with vision. " +
		"Supported formats: JPEG, PNG, GIF, WebP, BMP. " +
		"After calling this tool, describe or analyze the image in your next response."
}

func (t *LoadImageTool) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "Path to the local image file. Relative paths are resolved from workspace.",
			},
		},
		"required": []string{"path"},
	}
}

func (t *LoadImageTool) SetContext(channel, chatID string) {
	t.defaultChannel = channel
	t.defaultChatID = chatID
}

func (t *LoadImageTool) SetMediaStore(store media.MediaStore) {
	t.mediaStore = store
}

func (t *LoadImageTool) Execute(ctx context.Context, args map[string]any) *ToolResult {
	path, _ := args["path"].(string)
	if strings.TrimSpace(path) == "" {
		return ErrorResult("path is required")
	}

	// Prefer context-injected channel/chatID (set by ExecuteWithContext), fall back to SetContext values.
	channel := ToolChannel(ctx)
	if channel == "" {
		channel = t.defaultChannel
	}
	chatID := ToolChatID(ctx)
	if chatID == "" {
		chatID = t.defaultChatID
	}
	if channel == "" || chatID == "" {
		return ErrorResult("no target channel/chat available")
	}

	if t.mediaStore == nil {
		return ErrorResult("media store not configured")
	}

	resolved, err := validatePathWithAllowPaths(path, t.workspace, t.restrict, t.allowPaths)
	if err != nil {
		return ErrorResult(fmt.Sprintf("invalid path: %v", err))
	}

	info, err := os.Stat(resolved)
	if err != nil {
		return ErrorResult(fmt.Sprintf("file not found: %v", err))
	}
	if info.IsDir() {
		return ErrorResult("path is a directory, expected an image file")
	}
	if info.Size() > int64(t.maxFileSize) {
		return ErrorResult(fmt.Sprintf(
			"file too large: %d bytes (max %d bytes)", info.Size(), t.maxFileSize,
		))
	}

	// Detect MIME type — reuse the helper already in send_file.go
	mediaType := detectMediaType(resolved)
	if !strings.HasPrefix(mediaType, "image/") {
		return ErrorResult(fmt.Sprintf(
			"file does not appear to be an image (detected type: %s)", mediaType,
		))
	}

	filename := filepath.Base(resolved)
	storedPath, storedType, resized, err := normalizeVisionImage(resolved)
	if err != nil {
		return ErrorResult(fmt.Sprintf("failed to prepare image: %v", err))
	}
	cleanupPolicy := media.CleanupPolicyForgetOnly
	if resized {
		cleanupPolicy = media.CleanupPolicyDeleteOnCleanup
	}
	scope := fmt.Sprintf("tool:load_image:%s:%s", channel, chatID)

	ref, err := t.mediaStore.Store(storedPath, media.MediaMeta{
		Filename:      filename,
		ContentType:   storedType,
		Source:        "tool:load_image",
		CleanupPolicy: cleanupPolicy,
	}, scope)
	if err != nil {
		if resized {
			_ = os.Remove(storedPath)
		}
		return ErrorResult(fmt.Sprintf("failed to register image in media store: %v", err))
	}

	// Build the tool result text. The media:// ref in Media will be picked
	// up by resolveMediaRefs in agent_media.go and base64-encoded for tool
	// result messages (role="tool"), so the LLM can see the image content.
	msg := fmt.Sprintf("Image loaded: %s\n[image: photo]", filename)

	return &ToolResult{
		ForLLM:  msg,
		ForUser: fmt.Sprintf("Loaded image: %s", filename),
		// Media refs inside ForLLM are resolved by resolveMediaRefs in the
		// agent loop before the next LLM call. Do NOT use MediaResult here —
		// that would send the file to the user channel instead.
		Media: []string{ref},
	}
}

func normalizeVisionImage(path string) (storedPath, contentType string, resized bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", false, err
	}
	defer f.Close()

	cfg, format, err := image.DecodeConfig(f)
	if err != nil {
		return "", "", false, fmt.Errorf("decode dimensions: %w", err)
	}
	if cfg.Width <= maxVisionImageDimension && cfg.Height <= maxVisionImageDimension {
		return path, "image/" + format, false, nil
	}
	if _, err := f.Seek(0, 0); err != nil {
		return "", "", false, err
	}
	src, _, err := image.Decode(f)
	if err != nil {
		return "", "", false, fmt.Errorf("decode pixels: %w", err)
	}

	scale := float64(maxVisionImageDimension) / float64(max(cfg.Width, cfg.Height))
	width := max(1, int(float64(cfg.Width)*scale))
	height := max(1, int(float64(cfg.Height)*scale))
	dst := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Over, nil)

	tmp, err := os.CreateTemp("", "picoclaw-load-image-*.jpg")
	if err != nil {
		return "", "", false, err
	}
	tmpPath := tmp.Name()
	if err := jpeg.Encode(tmp, dst, &jpeg.Options{Quality: 90}); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return "", "", false, fmt.Errorf("encode resized image: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return "", "", false, err
	}
	return tmpPath, "image/jpeg", true, nil
}
