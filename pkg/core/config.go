package core

// Config represents the configuration for a transformer model
type Config struct {
	VocabSize    int
	EmbeddingDim int
	NumLayers    int // Applies to both encoder and decoder if NumDecoderLayers is 0
	NumDecoderLayers int // If > 0, specifies decoder layers, else NumLayers is used
	NumHeads     int
	FFNHiddenDim int     // Hidden dimension for standard FFN layers
	MaxLen       int
	DropoutRate  float64 // General dropout rate
	ActivationFuncName string // e.g., "gelu", "relu" for standard FFN, used to map to actual functions
	ActivationFunc func(interface{}) (interface{}, error) // Placeholder for actual function type from autodiff

	UseCrossLayerParameterSharing bool

	// MoE Specific configurations
	UseMoE bool    // Whether to use MoE layers instead of standard FFNs
	MoENumExperts int     // Number of experts in each MoE layer
	MoEHiddenDim  int     // Hidden dimension of each expert's FFN (if different from FFNHiddenDim)
	MoETopK       int     // Number of experts to route each token to
	MoECapacityFactor float64 // Factor to determine expert capacity
	MoENoisyRouting bool    // Whether to use noisy top-k routing
	MoEActivationName string  // e.g., "gelu", "relu" for MoE expert FFNs

	// These were previously in TensorFineTuningConfig, moving them here for model structure
	// Or they can be duplicated if fine-tuning needs to override model defaults.
	// For now, assuming model config might store these if they are intrinsic to MoE layer behavior.
	// However, loss coeffs are typically training-time, so better in TensorFineTuningConfig.
	// Let's keep them in TensorFineTuningConfig as per original plan.
	// MoERouterZLossCoeff     float64
	// MoELoadBalanceLossCoeff float64
}


// NewDefaultConfig creates a new configuration with default values
func NewDefaultConfig() *Config {
	return &Config{
		VocabSize:    10000,
		EmbeddingDim: 512,
		NumLayers:    6,
		NumHeads:     8,
		FFNHiddenDim: 2048,
		MaxLen:       512,
		DropoutRate:  0.1,
		ActivationFuncName: "gelu", // Default FFN activation
		UseCrossLayerParameterSharing: false,

		// MoE Defaults (sensible initial values)
		UseMoE:           false, // MoE disabled by default
		MoENumExperts:    8,
		MoEHiddenDim:     0,    // If 0, will default to e.g. EmbeddingDim * 4 or FFNHiddenDim in MoELayer/Expert
		MoETopK:          2,
		MoECapacityFactor:1.25,
		MoENoisyRouting:  true,
		MoEActivationName:"gelu",
	}
}

// NewConfig creates a new configuration with specified values (expanded)
// This constructor is getting very long. Consider using functional options or a builder pattern if it grows further.
func NewConfig(vocabSize, embeddingDim, numLayers, numDecoderLayers, numHeads, ffnHiddenDim, maxLen int, dropoutRate float64, activationName string, useCrossLayerSharing bool,
	useMoE bool, moeNumExperts, moeHiddenDim, moeTopK int, moeCapacityFactor float64, moeNoisyRouting bool, moeActivationName string) *Config {
	return &Config{
		VocabSize:    vocabSize,
		EmbeddingDim: embeddingDim,
		NumLayers:    numLayers,
		NumDecoderLayers: numDecoderLayers,
		NumHeads:     numHeads,
		FFNHiddenDim: ffnHiddenDim,
		MaxLen:       maxLen,
		DropoutRate:  dropoutRate,
		ActivationFuncName: activationName,
		UseCrossLayerParameterSharing: useCrossLayerSharing,
		UseMoE: useMoE,
		MoENumExperts: moeNumExperts,
		MoEHiddenDim: moeHiddenDim,
		MoETopK: moeTopK,
		MoECapacityFactor: moeCapacityFactor,
		MoENoisyRouting: moeNoisyRouting,
		MoEActivationName: moeActivationName,
	}
}
