package transformer

// Config represents the configuration for a transformer model
type Config struct {
	VocabSize    int
	EmbeddingDim int
	NumLayers    int
	NumHeads     int
	FFNHiddenDim int
	MaxLen       int
	UseCrossLayerParameterSharing bool
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
		UseCrossLayerParameterSharing: false, // Default to no sharing
	}
}

// NewConfig creates a new configuration with specified values
func NewConfig(vocabSize, embeddingDim, numLayers, numHeads, ffnHiddenDim, maxLen int, useCrossLayerParameterSharing bool) *Config {
	return &Config{
		VocabSize:    vocabSize,
		EmbeddingDim: embeddingDim,
		NumLayers:    numLayers,
		NumHeads:     numHeads,
		FFNHiddenDim: ffnHiddenDim,
		MaxLen:       maxLen,
		UseCrossLayerParameterSharing: useCrossLayerParameterSharing,
	}
}
