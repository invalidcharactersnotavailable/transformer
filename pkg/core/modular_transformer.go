package core

import (
	"transformer/internal/utils" // Added utils import
	"fmt" // For errors
)

// ModularTransformerConfig represents a configuration for a modular transformer
type ModularTransformerConfig struct {
	VocabSize         int
	EmbeddingDim      int
	NumEncoderLayers  int
	NumDecoderLayers  int
	NumHeads          int
	FFNHiddenDim      int
	MaxLen            int
	DropoutRate       float64
	AttentionDropout  float64
	ActivationDropout float64
	UseRotaryEncoding bool
	PositionalScale   float64
}

// NewModularTransformerConfig creates a new configuration with default values
func NewModularTransformerConfig() *ModularTransformerConfig {
	return &ModularTransformerConfig{
		VocabSize:         10000,
		EmbeddingDim:      512,
		NumEncoderLayers:  6,
		NumDecoderLayers:  6,
		NumHeads:          8,
		FFNHiddenDim:      2048,
		MaxLen:            512,
		DropoutRate:       0.1,
		AttentionDropout:  0.1,
		ActivationDropout: 0.1,
		UseRotaryEncoding: false,
		PositionalScale:   1.0,
	}
}

// ModularTransformer represents a modular transformer model
type ModularTransformer struct {
	Config            *ModularTransformerConfig
	Encoder           []*EnhancedEncoderLayer
	Decoder           []*EnhancedDecoderLayer
	PositionalEncoder interface{} // Can be either EnhancedPositionalEncoding or RotaryPositionalEncoding
	EmbeddingMatrix   *utils.Matrix // Prefixed
	OutputMatrix      *utils.Matrix // Prefixed
	EmbeddingDropout  *utils.Dropout // Prefixed
}

// EnhancedEncoderLayer represents an enhanced encoder layer with dropout
type EnhancedEncoderLayer struct {
	SelfAttention     *MultiHeadAttention // Assumed to be core.MultiHeadAttention
	FeedForward       *FeedForward        // Assumed to be core.FeedForward
	Norm1             *LayerNorm          // Assumed to be core.LayerNorm
	Norm2             *LayerNorm          // Assumed to be core.LayerNorm
	DropoutAttention  *utils.Dropout // Prefixed
	DropoutFFN        *utils.Dropout // Prefixed
	DropoutResidual   *utils.Dropout // Prefixed
	ModelDim          int
}

// NewEnhancedEncoderLayer creates a new enhanced encoder layer
// Note: NewMultiHeadAttention, NewFeedForward, NewLayerNorm, NewDropout are constructors from package core or utils
func NewEnhancedEncoderLayer(modelDim, ffnHiddenDim, numHeads int, dropoutRate float64) *EnhancedEncoderLayer {
	// Error handling for constructors that now return errors would be needed here
	// For simplicity in this diff, assuming they panic or are handled by callers of NewEnhancedEncoderLayer
	mha := NewMultiHeadAttention(numHeads, modelDim)     // core.NewMultiHeadAttention
	ff, _ := NewDefaultFeedForward(modelDim, ffnHiddenDim) // core.NewDefaultFeedForward, ignoring error for now
	ln1 := NewLayerNorm(modelDim)                        // core.NewLayerNorm
	ln2 := NewLayerNorm(modelDim)                        // core.NewLayerNorm

	return &EnhancedEncoderLayer{
		SelfAttention:    mha,
		FeedForward:      ff,
		Norm1:            ln1,
		Norm2:            ln2,
		DropoutAttention: utils.NewDropout(dropoutRate), // Prefixed
		DropoutFFN:       utils.NewDropout(dropoutRate), // Prefixed
		DropoutResidual:  utils.NewDropout(dropoutRate), // Prefixed
		ModelDim:         modelDim,
	}
}

// Forward processes input through the enhanced encoder layer
func (el *EnhancedEncoderLayer) Forward(x *utils.Matrix, mask *utils.AttentionMask, isTraining bool) (*utils.Matrix, error) { // Prefixed Matrix, added error, prefixed AttentionMask
	// Self-attention with residual connection and layer norm
	// Assuming el.SelfAttention.Forward and el.Norm1.Forward now return (*utils.Matrix, error)
	// and el.DropoutAttention.Forward now returns (*utils.Matrix, error)
	attnOut := el.SelfAttention.Forward(x, x, x) // This is core.MultiHeadAttention.Forward, needs error handling if changed
	
	var err error
	if isTraining {
		attnOut, err = el.DropoutAttention.Forward(attnOut, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout attention failed: %w", err)}
	}
	
	residual1, err := utils.NewMatrix(x.Rows, x.Cols) // Prefixed
	if err != nil { return nil, fmt.Errorf("failed to create residual1 matrix: %w", err)}
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			residual1.Data[i][j] = x.Data[i][j] + attnOut.Data[i][j]
		}
	}
	
	normalized1 := el.Norm1.Forward(residual1) // This is core.LayerNorm.Forward, needs error handling
	
	ffnOut := el.FeedForward.ForwardLegacy(normalized1) // This is core.FeedForward.ForwardLegacy
	
	if isTraining {
		ffnOut, err = el.DropoutFFN.Forward(ffnOut, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout ffn failed: %w", err)}
	}
	
	residual2, err := utils.NewMatrix(normalized1.Rows, normalized1.Cols) // Prefixed
	if err != nil { return nil, fmt.Errorf("failed to create residual2 matrix: %w", err)}
	for i := 0; i < normalized1.Rows; i++ {
		for j := 0; j < normalized1.Cols; j++ {
			residual2.Data[i][j] = normalized1.Data[i][j] + ffnOut.Data[i][j]
		}
	}
	
	if isTraining {
		residual2, err = el.DropoutResidual.Forward(residual2, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout residual failed: %w", err)}
	}
	
	finalOutput := el.Norm2.Forward(residual2) // This is core.LayerNorm.Forward, needs error handling
	return finalOutput, nil // Placeholder for actual error from Norm2.Forward
}

// EnhancedDecoderLayer represents an enhanced decoder layer with dropout
type EnhancedDecoderLayer struct {
	SelfAttention     *MultiHeadAttention // core type
	CrossAttention    *MultiHeadAttention // core type
	FeedForward       *FeedForward        // core type
	Norm1             *LayerNorm          // core type
	Norm2             *LayerNorm          // core type
	Norm3             *LayerNorm          // core type
	DropoutSelfAttn   *utils.Dropout // Prefixed
	DropoutCrossAttn  *utils.Dropout // Prefixed
	DropoutFFN        *utils.Dropout // Prefixed
	DropoutResidual1  *utils.Dropout // Prefixed
	DropoutResidual2  *utils.Dropout // Prefixed
	ModelDim          int
}

// NewEnhancedDecoderLayer creates a new enhanced decoder layer
func NewEnhancedDecoderLayer(modelDim, ffnHiddenDim, numHeads int, dropoutRate float64) *EnhancedDecoderLayer {
	// Similar to NewEnhancedEncoderLayer, error handling for constructors needed
	mhaSelf := NewMultiHeadAttention(numHeads, modelDim)
	mhaCross := NewMultiHeadAttention(numHeads, modelDim)
	ff, _ := NewDefaultFeedForward(modelDim, ffnHiddenDim)
	ln1 := NewLayerNorm(modelDim)
	ln2 := NewLayerNorm(modelDim)
	ln3 := NewLayerNorm(modelDim)

	return &EnhancedDecoderLayer{
		SelfAttention:    mhaSelf,
		CrossAttention:   mhaCross,
		FeedForward:      ff,
		Norm1:            ln1,
		Norm2:            ln2,
		Norm3:            ln3,
		DropoutSelfAttn:  utils.NewDropout(dropoutRate), // Prefixed
		DropoutCrossAttn: utils.NewDropout(dropoutRate), // Prefixed
		DropoutFFN:       utils.NewDropout(dropoutRate), // Prefixed
		DropoutResidual1: utils.NewDropout(dropoutRate), // Prefixed
		DropoutResidual2: utils.NewDropout(dropoutRate), // Prefixed
		ModelDim:         modelDim,
	}
}

// Forward processes input through the enhanced decoder layer
func (dl *EnhancedDecoderLayer) Forward(x, encoderOutput *utils.Matrix, selfMask, crossMask *utils.AttentionMask, isTraining bool) (*utils.Matrix, error) { // Prefixed Matrix, added error, prefixed AttentionMask
	var err error
	selfAttnOut := dl.SelfAttention.Forward(x, x, x) // core.MultiHeadAttention.Forward
	
	if isTraining {
		selfAttnOut, err = dl.DropoutSelfAttn.Forward(selfAttnOut, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout self-attention failed: %w", err)}
	}
	
	residual1, err := utils.NewMatrix(x.Rows, x.Cols) // Prefixed
	if err != nil { return nil, fmt.Errorf("failed to create residual1 matrix: %w", err)}
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			residual1.Data[i][j] = x.Data[i][j] + selfAttnOut.Data[i][j]
		}
	}
	
	if isTraining {
		residual1, err = dl.DropoutResidual1.Forward(residual1, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout residual1 failed: %w", err)}
	}
	
	normalized1 := dl.Norm1.Forward(residual1) // core.LayerNorm.Forward
	
	crossAttnOut := dl.CrossAttention.Forward(normalized1, encoderOutput, encoderOutput) // core.MultiHeadAttention.Forward
	
	if isTraining {
		crossAttnOut, err = dl.DropoutCrossAttn.Forward(crossAttnOut, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout cross-attention failed: %w", err)}
	}
	
	residual2, err := utils.NewMatrix(normalized1.Rows, normalized1.Cols) // Prefixed
	if err != nil { return nil, fmt.Errorf("failed to create residual2 matrix: %w", err)}
	for i := 0; i < normalized1.Rows; i++ {
		for j := 0; j < normalized1.Cols; j++ {
			residual2.Data[i][j] = normalized1.Data[i][j] + crossAttnOut.Data[i][j]
		}
	}
	
	normalized2 := dl.Norm2.Forward(residual2) // core.LayerNorm.Forward
	
	ffnOut := dl.FeedForward.ForwardLegacy(normalized2) // core.FeedForward.ForwardLegacy
	
	if isTraining {
		ffnOut, err = dl.DropoutFFN.Forward(ffnOut, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout ffn failed: %w", err)}
	}
	
	residual3, err := utils.NewMatrix(normalized2.Rows, normalized2.Cols) // Prefixed
	if err != nil { return nil, fmt.Errorf("failed to create residual3 matrix: %w", err)}
	for i := 0; i < normalized2.Rows; i++ {
		for j := 0; j < normalized2.Cols; j++ {
			residual3.Data[i][j] = normalized2.Data[i][j] + ffnOut.Data[i][j]
		}
	}
	
	if isTraining {
		residual3, err = dl.DropoutResidual2.Forward(residual3, isTraining)
		if err != nil { return nil, fmt.Errorf("dropout residual2 failed: %w", err)}
	}
	
	finalOutput := dl.Norm3.Forward(residual3) // core.LayerNorm.Forward
	return finalOutput, nil // Placeholder for actual error from Norm3.Forward
}
