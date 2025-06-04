package core

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	// Assuming autodiff and moe might be needed for type references if config structs are complex
	// For now, just basic types.
)

// HyperParameters represents all configurable parameters for transformer models
type HyperParameters struct {
	// Model architecture
	ModelType          string  `json:"model_type"` // e.g., "transformer", "moe_transformer"
	VocabSize          int     `json:"vocab_size"`
	EmbeddingDim       int     `json:"embedding_dim"`
	NumEncoderLayers   int     `json:"num_encoder_layers"` // Used for Config.NumLayers
	NumDecoderLayers   int     `json:"num_decoder_layers"` // Used for Config.NumDecoderLayers
	NumHeads           int     `json:"num_heads"`
	FFNHiddenDim       int     `json:"ffn_hidden_dim"`    // For standard FFN
	DropoutRate        float64 `json:"dropout_rate"`
	ActivationFuncName string  `json:"activation_func_name"` // For standard FFN, e.g. "gelu", "relu"
	UseCrossLayerParameterSharing bool `json:"use_cross_layer_parameter_sharing"`


	// MoE Specific Configurations
	UseMoE            bool    `json:"use_moe"`
	MoENumExperts     int     `json:"moe_num_experts"`
	MoEHiddenDim      int     `json:"moe_expert_hidden_dim"` // Expert FFN hidden dim
	MoETopK           int     `json:"moe_top_k"`
	MoECapacityFactor float64 `json:"moe_capacity_factor"`
	MoENoisyRouting   bool    `json:"moe_noisy_routing"`
	MoEActivationName string  `json:"moe_activation_name"`   // e.g., "gelu", "relu" for experts

	// MoE Loss Coefficients (typically part of training/fine-tuning config)
	MoERouterZLossCoeff     float64 `json:"moe_router_z_loss_coeff"`
	MoELoadBalanceLossCoeff float64 `json:"moe_load_balance_loss_coeff"`
	
	// Context and sequence handling
	MaxContextLength   int     `json:"max_context_length"` // Might be same as MaxLen for model config
	MaxSequenceLength  int     `json:"max_sequence_length"`// Used for Config.MaxLen
	ContextChunkSize   int     `json:"context_chunk_size"`
	ContextOverlap     int     `json:"context_overlap"`
	
	// Generation parameters (can be a separate struct if grows too large)
	Temperature        float64 `json:"temperature"`
	TopK               int     `json:"top_k"`
	TopP               float64 `json:"top_p"`
	RepetitionPenalty  float64 `json:"repetition_penalty"`
	LengthPenalty      float64 `json:"length_penalty"`
	BeamSize           int     `json:"beam_size"`
	
	// Training parameters (can be a separate struct)
	LearningRate       float64 `json:"learning_rate"`
	WeightDecay        float64 `json:"weight_decay"`
	WarmupSteps        int     `json:"warmup_steps"`
	BatchSize          int     `json:"batch_size"`
	NumEpochs          int     `json:"num_epochs"` // For fine-tuning config
	GradientClipValue  float64 `json:"gradient_clip_value"`
	OptimizerName      string  `json:"optimizer_name"` // e.g. "adamw", "sgd" for fine-tuning
	SaveFrequency      int     `json:"save_frequency"` // For fine-tuning
	TokenizerVocabFile string  `json:"tokenizer_vocab_file"` // For fine-tuning
	
	// Multimodal parameters
	ImageDim           int     `json:"image_dim"`
	JointDim           int     `json:"joint_dim"`
	MaxImagePatches    int     `json:"max_image_patches"`
	
	// Advanced options (could be model or training specific)
	UseRotaryEncoding  bool    `json:"use_rotary_encoding"` // Model config
	UseGELU            bool    `json:"use_gelu"`            // Model config (legacy if ActivationFuncName is used)
	UseStreaming       bool    `json:"use_streaming"`       // Generation/Inference config
	UseKVCache         bool    `json:"use_kv_cache"`        // Generation/Inference config
}

// NewDefaultHyperParameters creates default hyperparameters
func NewDefaultHyperParameters() *HyperParameters {
	return &HyperParameters{
		ModelType:          "transformer", VocabSize:          50257, EmbeddingDim:       768,
		NumEncoderLayers:   6, NumDecoderLayers:   6, NumHeads:           12,
		FFNHiddenDim:       3072, DropoutRate:        0.1, ActivationFuncName: "gelu",
		UseCrossLayerParameterSharing: false,

		UseMoE:            false, MoENumExperts:     8, MoEHiddenDim:      0, // 0 means use default based on EmbeddingDim/FFNHiddenDim
		MoETopK:           2, MoECapacityFactor: 1.25, MoENoisyRouting:   true,
		MoEActivationName: "gelu", MoERouterZLossCoeff: 0.01, MoELoadBalanceLossCoeff: 0.01,

		MaxContextLength:   4096, MaxSequenceLength:  1024, ContextChunkSize:   512, ContextOverlap:     128,
		Temperature:        0.7, TopK:               50, TopP:               0.9, RepetitionPenalty:  1.0, LengthPenalty:      1.0, BeamSize:           1,
		LearningRate:       5e-5, WeightDecay:        0.01, WarmupSteps:        100, BatchSize:          16, NumEpochs: 3,
		GradientClipValue:  1.0, OptimizerName: "adamw", SaveFrequency: 1, TokenizerVocabFile: "",
		ImageDim:           768, JointDim:           768, MaxImagePatches:    196,
		UseRotaryEncoding:  false, UseGELU:            true, UseStreaming:       false, UseKVCache:         true,
	}
}

// SaveHyperParameters saves hyperparameters to a JSON file
func SaveHyperParameters(params *HyperParameters, filePath string) error {
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil { return fmt.Errorf("create dir: %w", err) }
	data, err := json.MarshalIndent(params, "", "  "); if err != nil { return fmt.Errorf("marshal: %w", err) }
	return ioutil.WriteFile(filePath, data, 0644)
}

// LoadHyperParameters loads hyperparameters from a JSON file
func LoadHyperParameters(filePath string) (*HyperParameters, error) {
	data, err := ioutil.ReadFile(filePath); if err != nil { return nil, fmt.Errorf("read file: %w", err) }
	var params HyperParameters; if err := json.Unmarshal(data, &params); err != nil { return nil, fmt.Errorf("unmarshal: %w", err) }
	return &params, nil
}


// ContextManager (remains unchanged for this subtask)
type ContextManager struct { MaxContextLength, ChunkSize, Overlap int; CurrentContext []int; mutex sync.Mutex }
func NewContextManager(maxCtx, chunkSize, overlap int) *ContextManager { return &ContextManager{MaxContextLength:maxCtx,ChunkSize:chunkSize,Overlap:overlap}}
// ... (AddTokens, GetCurrentWindow, GetChunkedWindows, Clear remain unchanged)


// HyperParameterManager manages hyperparameters for different components
type HyperParameterManager struct {
	Params *HyperParameters
}
func NewHyperParameterManager(params *HyperParameters) *HyperParameterManager {
	if params == nil { params = NewDefaultHyperParameters() }
	return &HyperParameterManager{Params: params}
}

// CreateModelConfig creates a model configuration (core.Config) from hyperparameters
func (hpm *HyperParameterManager) CreateModelConfig() *Config { // Assuming Config is from pkg/core
	// ActivationFunc mapping would happen here or in NewTransformerWithTensors
	// For now, just pass the name.
	return &Config{
		VocabSize:    hpm.Params.VocabSize, EmbeddingDim: hpm.Params.EmbeddingDim,
		NumLayers:    hpm.Params.NumEncoderLayers, NumDecoderLayers: hpm.Params.NumDecoderLayers,
		NumHeads:     hpm.Params.NumHeads, FFNHiddenDim: hpm.Params.FFNHiddenDim,
		MaxLen:       hpm.Params.MaxSequenceLength, DropoutRate:  hpm.Params.DropoutRate,
		ActivationFuncName: hpm.Params.ActivationFuncName,
		UseCrossLayerParameterSharing: hpm.Params.UseCrossLayerParameterSharing,

		UseMoE:            hpm.Params.UseMoE,
		MoENumExperts:     hpm.Params.MoENumExperts,
		MoEHiddenDim:      hpm.Params.MoEHiddenDim,
		MoETopK:           hpm.Params.MoETopK,
		MoECapacityFactor: hpm.Params.MoECapacityFactor,
		MoENoisyRouting:   hpm.Params.MoENoisyRouting,
		MoEActivationName: hpm.Params.MoEActivationName,
		// MoE Loss coefficients are part of training config, not model structure config.
	}
}

// CreateFineTuningConfig creates a fine-tuning configuration (autodiff.TensorFineTuningConfig)
// This function is illustrative; the actual TensorFineTuningConfig might be in autodiff package.
// For now, this shows how HPs would flow to it.
func (hpm *HyperParameterManager) CreateFineTuningConfig() interface{} { // Return interface{} or specific type
	// This is a conceptual placeholder for where TensorFineTuningConfig would be created.
	// Let's assume TensorFineTuningConfig is defined in `autodiff` package.
	// We would need to import `autodiff` and use its constructor.
	// For now, just returning a map to show the fields.
	return map[string]interface{}{
		"LearningRate":      hpm.Params.LearningRate,
		"BatchSize":         hpm.Params.BatchSize,
		"NumEpochs":         hpm.Params.NumEpochs,
		"WarmupSteps":       hpm.Params.WarmupSteps,
		"WeightDecay":       hpm.Params.WeightDecay,
		"ClipGradNorm":      hpm.Params.GradientClipValue,
		"Optimizer":         hpm.Params.OptimizerName,
		"SaveFrequency":     hpm.Params.SaveFrequency,
		"TokenizerVocabFile":hpm.Params.TokenizerVocabFile,
		"MaxSeqLen":         hpm.Params.MaxSequenceLength, // Often needed by tokenizer during fine-tuning data prep
		// MoE Loss Coefficients for training
		"MoERouterZLossCoeff":     hpm.Params.MoERouterZLossCoeff,
		"MoELoadBalanceLossCoeff": hpm.Params.MoELoadBalanceLossCoeff,
	}
}

// ... (Other methods like CreateGenerationConfig, CreateMultimodalConfig, UpdateFromJSON, GetJSON remain unchanged or need similar updates)
func (hpm *HyperParameterManager) UpdateFromJSON(jsonStr string) error {
	var params HyperParameters; if err := json.Unmarshal([]byte(jsonStr), &params); err != nil { return fmt.Errorf("unmarshal: %w", err) }
	hpm.Params = &params; return nil
}
func (hpm *HyperParameterManager) GetJSON() (string, error) {
	data, err := json.MarshalIndent(hpm.Params, "", "  "); if err != nil { return "", fmt.Errorf("marshal: %w", err) }
	return string(data), nil
}

// AddTokens, GetCurrentWindow, GetChunkedWindows, Clear for ContextManager (if needed by HyperParameterManager methods)
func (cm *ContextManager) AddTokens(tokens []int) { cm.mutex.Lock(); defer cm.mutex.Unlock(); cm.CurrentContext = append(cm.CurrentContext, tokens...); if len(cm.CurrentContext) > cm.MaxContextLength { excess := len(cm.CurrentContext) - cm.MaxContextLength; cm.CurrentContext = cm.CurrentContext[excess:] } }
func (cm *ContextManager) GetCurrentWindow() []int { cm.mutex.Lock(); defer cm.mutex.Unlock(); if len(cm.CurrentContext) <= cm.ChunkSize {拷贝 := make([]int, len(cm.CurrentContext)); copy(拷贝, cm.CurrentContext); return 拷贝}; start := len(cm.CurrentContext) - cm.ChunkSize; if start < 0 { start = 0 }; 拷贝 := make([]int, len(cm.CurrentContext)-start); copy(拷贝, cm.CurrentContext[start:]); return 拷贝 }
func (cm *ContextManager) Clear() { cm.mutex.Lock(); defer cm.mutex.Unlock(); cm.CurrentContext = []int{} }
