// Package core contains the core components of the transformer architecture
package core

// Import necessary packages
// import (
//	"transformer/internal/utils" // Temporarily removed to break import cycle
// )

// Re-export key types and functions
type Config struct {
	VocabSize    int
	EmbeddingDim int
	NumLayers    int
	NumHeads     int
	FFNHiddenDim int
	MaxLen       int
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
	}
}

// NewConfig creates a new configuration with specified values
func NewConfig(vocabSize, embeddingDim, numLayers, numHeads, ffnHiddenDim, maxLen int) *Config {
	return &Config{
		VocabSize:    vocabSize,
		EmbeddingDim: embeddingDim,
		NumLayers:    numLayers,
		NumHeads:     numHeads,
		FFNHiddenDim: ffnHiddenDim,
		MaxLen:       maxLen,
	}
}
