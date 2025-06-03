package transformer

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
)

// HyperParameters represents all configurable parameters for transformer models
type HyperParameters struct {
	// Model architecture
	ModelType          string  `json:"model_type"`
	VocabSize          int     `json:"vocab_size"`
	EmbeddingDim       int     `json:"embedding_dim"`
	NumEncoderLayers   int     `json:"num_encoder_layers"`
	NumDecoderLayers   int     `json:"num_decoder_layers"`
	NumHeads           int     `json:"num_heads"`
	FFNHiddenDim       int     `json:"ffn_hidden_dim"`
	DropoutRate        float64 `json:"dropout_rate"`
	AttentionDropout   float64 `json:"attention_dropout"`
	ActivationDropout  float64 `json:"activation_dropout"`
	
	// Context and sequence handling
	MaxContextLength   int     `json:"max_context_length"`
	MaxSequenceLength  int     `json:"max_sequence_length"`
	ContextChunkSize   int     `json:"context_chunk_size"`
	ContextOverlap     int     `json:"context_overlap"`
	
	// Generation parameters
	Temperature        float64 `json:"temperature"`
	TopK               int     `json:"top_k"`
	TopP               float64 `json:"top_p"`
	RepetitionPenalty  float64 `json:"repetition_penalty"`
	LengthPenalty      float64 `json:"length_penalty"`
	BeamSize           int     `json:"beam_size"`
	
	// Training parameters
	LearningRate       float64 `json:"learning_rate"`
	WeightDecay        float64 `json:"weight_decay"`
	WarmupSteps        int     `json:"warmup_steps"`
	BatchSize          int     `json:"batch_size"`
	GradientClipValue  float64 `json:"gradient_clip_value"`
	
	// Multimodal parameters
	ImageDim           int     `json:"image_dim"`
	JointDim           int     `json:"joint_dim"`
	MaxImagePatches    int     `json:"max_image_patches"`
	
	// Advanced options
	UseRotaryEncoding  bool    `json:"use_rotary_encoding"`
	UseGELU            bool    `json:"use_gelu"`
	UseStreaming       bool    `json:"use_streaming"`
	UseKVCache         bool    `json:"use_kv_cache"`
}

// NewDefaultHyperParameters creates default hyperparameters
func NewDefaultHyperParameters() *HyperParameters {
	return &HyperParameters{
		// Model architecture
		ModelType:          "transformer",
		VocabSize:          50000,
		EmbeddingDim:       768,
		NumEncoderLayers:   6,
		NumDecoderLayers:   6,
		NumHeads:           12,
		FFNHiddenDim:       3072,
		DropoutRate:        0.1,
		AttentionDropout:   0.1,
		ActivationDropout:  0.1,
		
		// Context and sequence handling
		MaxContextLength:   4096,
		MaxSequenceLength:  1024,
		ContextChunkSize:   512,
		ContextOverlap:     128,
		
		// Generation parameters
		Temperature:        1.0,
		TopK:               50,
		TopP:               0.9,
		RepetitionPenalty:  1.0,
		LengthPenalty:      1.0,
		BeamSize:           1,
		
		// Training parameters
		LearningRate:       1e-4,
		WeightDecay:        0.01,
		WarmupSteps:        1000,
		BatchSize:          16,
		GradientClipValue:  1.0,
		
		// Multimodal parameters
		ImageDim:           768,
		JointDim:           768,
		MaxImagePatches:    196,
		
		// Advanced options
		UseRotaryEncoding:  false,
		UseGELU:            true,
		UseStreaming:       false,
		UseKVCache:         true,
	}
}

// SaveHyperParameters saves hyperparameters to a JSON file
func SaveHyperParameters(params *HyperParameters, filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}
	
	// Marshal to JSON
	data, err := json.MarshalIndent(params, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal hyperparameters: %v", err)
	}
	
	// Write to file
	if err := ioutil.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write hyperparameters: %v", err)
	}
	
	return nil
}

// LoadHyperParameters loads hyperparameters from a JSON file
func LoadHyperParameters(filePath string) (*HyperParameters, error) {
	// Read file
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read hyperparameters: %v", err)
	}
	
	// Unmarshal JSON
	var params HyperParameters
	if err := json.Unmarshal(data, &params); err != nil {
		return nil, fmt.Errorf("failed to unmarshal hyperparameters: %v", err)
	}
	
	return &params, nil
}

// ContextManager handles context windowing for long sequences
type ContextManager struct {
	MaxContextLength  int
	ChunkSize         int
	Overlap           int
	CurrentContext    []int
	mutex             sync.Mutex
}

// NewContextManager creates a new context manager
func NewContextManager(maxContextLength, chunkSize, overlap int) *ContextManager {
	return &ContextManager{
		MaxContextLength: maxContextLength,
		ChunkSize:        chunkSize,
		Overlap:          overlap,
		CurrentContext:   []int{},
	}
}

// AddTokens adds tokens to the context
func (cm *ContextManager) AddTokens(tokens []int) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	
	// Add new tokens
	cm.CurrentContext = append(cm.CurrentContext, tokens...)
	
	// Trim if exceeding max length
	if len(cm.CurrentContext) > cm.MaxContextLength {
		excess := len(cm.CurrentContext) - cm.MaxContextLength
		cm.CurrentContext = cm.CurrentContext[excess:]
	}
}

// GetCurrentWindow gets the current context window
func (cm *ContextManager) GetCurrentWindow() []int {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	
	// If context is smaller than chunk size, return all
	if len(cm.CurrentContext) <= cm.ChunkSize {
		result := make([]int, len(cm.CurrentContext))
		copy(result, cm.CurrentContext)
		return result
	}
	
	// Otherwise, return the last chunk
	start := len(cm.CurrentContext) - cm.ChunkSize
	if start < 0 {
		start = 0
	}
	
	result := make([]int, len(cm.CurrentContext)-start)
	copy(result, cm.CurrentContext[start:])
	return result
}

// GetChunkedWindows gets multiple overlapping windows for processing long contexts
func (cm *ContextManager) GetChunkedWindows() [][]int {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	
	// If context is smaller than chunk size, return single chunk
	if len(cm.CurrentContext) <= cm.ChunkSize {
		return [][]int{append([]int{}, cm.CurrentContext...)}
	}
	
	// Calculate number of chunks
	numChunks := (len(cm.CurrentContext) - cm.Overlap) / (cm.ChunkSize - cm.Overlap)
	if (len(cm.CurrentContext) - cm.Overlap) % (cm.ChunkSize - cm.Overlap) > 0 {
		numChunks++
	}
	
	// Create chunks
	chunks := make([][]int, numChunks)
	for i := 0; i < numChunks; i++ {
		start := i * (cm.ChunkSize - cm.Overlap)
		end := start + cm.ChunkSize
		if end > len(cm.CurrentContext) {
			end = len(cm.CurrentContext)
		}
		
		chunks[i] = make([]int, end-start)
		copy(chunks[i], cm.CurrentContext[start:end])
	}
	
	return chunks
}

// Clear clears the context
func (cm *ContextManager) Clear() {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()
	
	cm.CurrentContext = []int{}
}

// ContextStreamingGenerator handles streaming generation with context management
type ContextStreamingGenerator struct {
	Generator      *AdvancedGenerator
	ContextManager *ContextManager
	Tokens         []int
	IsRunning      bool
	mutex          sync.Mutex
}

// NewContextStreamingGenerator creates a new context streaming generator
func NewContextStreamingGenerator(generator *AdvancedGenerator, contextManager *ContextManager) *ContextStreamingGenerator {
	return &ContextStreamingGenerator{
		Generator:      generator,
		ContextManager: contextManager,
		Tokens:         []int{},
		IsRunning:      false,
	}
}

// StartGeneration starts streaming generation with context management
func (csg *ContextStreamingGenerator) StartGeneration(inputTokens []int, callback GenerationCallback) {
	csg.mutex.Lock()
	
	// Initialize with input tokens
	csg.Tokens = make([]int, len(inputTokens))
	copy(csg.Tokens, inputTokens)
	csg.IsRunning = true
	
	// Add to context manager
	csg.ContextManager.AddTokens(inputTokens)
	
	// Get current context window
	contextWindow := csg.ContextManager.GetCurrentWindow()
	
	csg.mutex.Unlock()
	
	// Start generation
	go func() {
		csg.Generator.Generate(contextWindow, func(tokens []int, isFinished bool) bool {
			csg.mutex.Lock()
			defer csg.mutex.Unlock()
			
			if !csg.IsRunning {
				return false
			}
			
			// Extract only new tokens (not in the context window)
			var newTokens []int
			if len(tokens) > len(contextWindow) {
				newTokens = tokens[len(contextWindow):]
				
				// Update tokens
				csg.Tokens = append(csg.Tokens, newTokens...)
				
				// Add to context manager
				csg.ContextManager.AddTokens(newTokens)
			}
			
			return callback(csg.Tokens, isFinished)
		})
	}()
}

// StopGeneration stops streaming generation
func (csg *ContextStreamingGenerator) StopGeneration() {
	csg.mutex.Lock()
	defer csg.mutex.Unlock()
	
	csg.IsRunning = false
}

// GetCurrentTokens gets the current tokens
func (csg *ContextStreamingGenerator) GetCurrentTokens() []int {
	csg.mutex.Lock()
	defer csg.mutex.Unlock()
	
	result := make([]int, len(csg.Tokens))
	copy(result, csg.Tokens)
	return result
}

// HyperParameterManager manages hyperparameters for different components
type HyperParameterManager struct {
	Params *HyperParameters
}

// NewHyperParameterManager creates a new hyperparameter manager
func NewHyperParameterManager(params *HyperParameters) *HyperParameterManager {
	if params == nil {
		params = NewDefaultHyperParameters()
	}
	
	return &HyperParameterManager{
		Params: params,
	}
}

// CreateModelConfig creates a model configuration from hyperparameters
func (hpm *HyperParameterManager) CreateModelConfig() *Config {
	return &Config{
		VocabSize:    hpm.Params.VocabSize,
		EmbeddingDim: hpm.Params.EmbeddingDim,
		NumLayers:    hpm.Params.NumEncoderLayers,
		NumHeads:     hpm.Params.NumHeads,
		FFNHiddenDim: hpm.Params.FFNHiddenDim,
		MaxLen:       hpm.Params.MaxSequenceLength,
	}
}

// CreateFineTuningConfig creates a fine-tuning configuration from hyperparameters
func (hpm *HyperParameterManager) CreateFineTuningConfig() *FineTuningConfig {
	return &FineTuningConfig{
		LearningRate:      hpm.Params.LearningRate,
		BatchSize:         hpm.Params.BatchSize,
		NumEpochs:         3, // Default value
		GradientClipValue: hpm.Params.GradientClipValue,
		WarmupSteps:       hpm.Params.WarmupSteps,
		WeightDecay:       hpm.Params.WeightDecay,
	}
}

// CreateGenerationConfig creates a generation configuration from hyperparameters
func (hpm *HyperParameterManager) CreateGenerationConfig() *GenerationConfig {
	return &GenerationConfig{
		MaxLength:         hpm.Params.MaxSequenceLength,
		MinLength:         1,
		Temperature:       hpm.Params.Temperature,
		TopK:              hpm.Params.TopK,
		TopP:              hpm.Params.TopP,
		RepetitionPenalty: hpm.Params.RepetitionPenalty,
		LengthPenalty:     hpm.Params.LengthPenalty,
		BeamSize:          hpm.Params.BeamSize,
		DoSample:          hpm.Params.Temperature > 0.0,
		UseStreaming:      hpm.Params.UseStreaming,
	}
}

// CreateMultimodalConfig creates a multimodal configuration from hyperparameters
func (hpm *HyperParameterManager) CreateMultimodalConfig() *MultimodalConfig {
	return &MultimodalConfig{
		TextDim:          hpm.Params.EmbeddingDim,
		ImageDim:         hpm.Params.ImageDim,
		JointDim:         hpm.Params.JointDim,
		NumEncoderLayers: hpm.Params.NumEncoderLayers,
		NumDecoderLayers: hpm.Params.NumDecoderLayers,
		NumHeads:         hpm.Params.NumHeads,
		FFNHiddenDim:     hpm.Params.FFNHiddenDim,
		MaxTextLen:       hpm.Params.MaxSequenceLength,
		MaxImagePatches:  hpm.Params.MaxImagePatches,
	}
}

// CreateContextManager creates a context manager from hyperparameters
func (hpm *HyperParameterManager) CreateContextManager() *ContextManager {
	return NewContextManager(
		hpm.Params.MaxContextLength,
		hpm.Params.ContextChunkSize,
		hpm.Params.ContextOverlap,
	)
}

// UpdateFromJSON updates hyperparameters from a JSON string
func (hpm *HyperParameterManager) UpdateFromJSON(jsonStr string) error {
	var params HyperParameters
	if err := json.Unmarshal([]byte(jsonStr), &params); err != nil {
		return fmt.Errorf("failed to unmarshal hyperparameters: %v", err)
	}
	
	hpm.Params = &params
	return nil
}

// GetJSON returns hyperparameters as a JSON string
func (hpm *HyperParameterManager) GetJSON() (string, error) {
	data, err := json.MarshalIndent(hpm.Params, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal hyperparameters: %v", err)
	}
	
	return string(data), nil
}
