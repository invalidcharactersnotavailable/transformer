package transformer

import (
	"fmt"
	"math"
)

// AttentionType represents the type of attention mechanism to use
type AttentionType int

const (
	// StandardAttention uses basic scaled dot-product attention
	StandardAttention AttentionType = iota
	// MaskedAttention uses attention with masking
	MaskedAttention
)

// AttentionConfig holds configuration options for creating an attention mechanism
type AttentionConfig struct {
	NumHeads     int
	ModelDim     int
	DropoutRate  float64
	AttentionType AttentionType
}

// MultiHeadAttention represents a configurable multi-head attention mechanism
type MultiHeadAttention struct {
	NumHeads        int
	ModelDim        int
	HeadDim         int
	QueryWeight     *Matrix
	KeyWeight       *Matrix
	ValueWeight     *Matrix
	OutputWeight    *Matrix
	AttentionDropout *Dropout
	DropoutRate     float64
	AttentionType   AttentionType
}

// NewMultiHeadAttention creates a new multi-head attention layer with the specified configuration
func NewMultiHeadAttention(config AttentionConfig) (*MultiHeadAttention, error) {
	if config.NumHeads <= 0 {
		return nil, fmt.Errorf("number of heads must be positive, got %d", config.NumHeads)
	}
	
	if config.ModelDim <= 0 {
		return nil, fmt.Errorf("model dimension must be positive, got %d", config.ModelDim)
	}
	
	if config.ModelDim % config.NumHeads != 0 {
		return nil, fmt.Errorf("model dimension (%d) must be divisible by number of heads (%d)", 
			config.ModelDim, config.NumHeads)
	}
	
	if config.DropoutRate < 0 || config.DropoutRate >= 1.0 {
		return nil, fmt.Errorf("dropout rate must be in range [0, 1), got %f", config.DropoutRate)
	}
	
	headDim := config.ModelDim / config.NumHeads
	
	queryWeight, err := NewRandomMatrix(config.ModelDim, config.ModelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create query weight matrix: %v", err)
	}
	
	keyWeight, err := NewRandomMatrix(config.ModelDim, config.ModelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create key weight matrix: %v", err)
	}
	
	valueWeight, err := NewRandomMatrix(config.ModelDim, config.ModelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create value weight matrix: %v", err)
	}
	
	outputWeight, err := NewRandomMatrix(config.ModelDim, config.ModelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create output weight matrix: %v", err)
	}
	
	return &MultiHeadAttention{
		NumHeads:        config.NumHeads,
		ModelDim:        config.ModelDim,
		HeadDim:         headDim,
		QueryWeight:     queryWeight,
		KeyWeight:       keyWeight,
		ValueWeight:     valueWeight,
		OutputWeight:    outputWeight,
		AttentionDropout: NewDropout(config.DropoutRate),
		DropoutRate:     config.DropoutRate,
		AttentionType:   config.AttentionType,
	}, nil
}

// NewStandardMultiHeadAttention creates a new standard multi-head attention layer
func NewStandardMultiHeadAttention(numHeads, modelDim int) (*MultiHeadAttention, error) {
	return NewMultiHeadAttention(AttentionConfig{
		NumHeads:     numHeads,
		ModelDim:     modelDim,
		DropoutRate:  0.0,
		AttentionType: StandardAttention,
	})
}

// NewAdvancedMultiHeadAttention creates a new advanced multi-head attention layer with dropout
func NewAdvancedMultiHeadAttention(numHeads, modelDim int, dropoutRate float64) (*MultiHeadAttention, error) {
	return NewMultiHeadAttention(AttentionConfig{
		NumHeads:     numHeads,
		ModelDim:     modelDim,
		DropoutRate:  dropoutRate,
		AttentionType: MaskedAttention,
	})
}

// AttentionMask represents a mask for attention weights
type AttentionMask struct {
	Mask *Matrix
}

// NewPaddingMask creates a new padding mask from an attention mask tensor
func NewPaddingMask(attentionMask *Matrix) (*AttentionMask, error) {
	if attentionMask == nil {
		return nil, fmt.Errorf("attention mask cannot be nil")
	}
	
	rows := attentionMask.Rows
	cols := attentionMask.Cols
	
	// Create a mask matrix of the same shape as the attention scores would be
	mask, err := NewMatrix(rows, rows)
	if err != nil {
		return nil, fmt.Errorf("failed to create mask matrix: %v", err)
	}
	
	// Initialize with ones (allow attention)
	for i := 0; i < rows; i++ {
		for j := 0; j < rows; j++ {
			mask.Data[i][j] = 1.0
		}
	}
	
	// Apply padding mask: if a token is padding (attention_mask[i] == 0),
	// then it should not be attended to
	for i := 0; i < rows; i++ {
		if i < cols && attentionMask.Data[0][i] == 0 {
			// This is a padding position, mask it in all rows
			for j := 0; j < rows; j++ {
				mask.Data[j][i] = 0.0
			}
		}
	}
	
	return &AttentionMask{
		Mask: mask,
	}, nil
}

// NewCausalMask creates a new causal (autoregressive) mask
func NewCausalMask(size int) (*AttentionMask, error) {
	if size <= 0 {
		return nil, fmt.Errorf("mask size must be positive, got %d", size)
	}
	
	mask, err := NewMatrix(size, size)
	if err != nil {
		return nil, fmt.Errorf("failed to create mask matrix: %v", err)
	}
	
	// Create lower triangular matrix (1s on and below diagonal, 0s above)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if j <= i {
				mask.Data[i][j] = 1.0
			} else {
				mask.Data[i][j] = 0.0
			}
		}
	}
	
	return &AttentionMask{
		Mask: mask,
	}, nil
}

// ApplyMask applies the mask to attention scores
func (am *AttentionMask) ApplyMask(scores *Matrix) (*Matrix, error) {
	if am == nil || am.Mask == nil {
		return scores, nil
	}
	
	if scores == nil {
		return nil, fmt.Errorf("scores matrix cannot be nil")
	}
	
	if scores.Rows != am.Mask.Rows || scores.Cols != am.Mask.Cols {
		return nil, fmt.Errorf("mask dimensions (%dx%d) don't match scores dimensions (%dx%d)",
			am.Mask.Rows, am.Mask.Cols, scores.Rows, scores.Cols)
	}
	
	result, err := NewMatrix(scores.Rows, scores.Cols)
	if err != nil {
		return nil, fmt.Errorf("failed to create result matrix: %v", err)
	}
	
	// Apply mask: where mask is 0, set score to negative infinity (or very negative number)
	for i := 0; i < scores.Rows; i++ {
		for j := 0; j < scores.Cols; j++ {
			if am.Mask.Data[i][j] == 0 {
				result.Data[i][j] = -1e9 // Approximation of negative infinity
			} else {
				result.Data[i][j] = scores.Data[i][j]
			}
		}
	}
	
	return result, nil
}

// Forward performs the multi-head attention operation
func (mha *MultiHeadAttention) Forward(query, key, value *Matrix, mask *AttentionMask, isTraining bool) (*Matrix, error) {
	if query == nil || key == nil || value == nil {
		return nil, fmt.Errorf("query, key, and value matrices cannot be nil")
	}
	
	if query.Cols != mha.ModelDim || key.Cols != mha.ModelDim || value.Cols != mha.ModelDim {
		return nil, fmt.Errorf("input dimensions don't match model dimension (%d): query(%d), key(%d), value(%d)",
			mha.ModelDim, query.Cols, key.Cols, value.Cols)
	}
	
	// Project inputs to query, key, value
	q, err := MatMul(query, mha.QueryWeight)
	if err != nil {
		return nil, fmt.Errorf("failed to project query: %v", err)
	}
	
	k, err := MatMul(key, mha.KeyWeight)
	if err != nil {
		return nil, fmt.Errorf("failed to project key: %v", err)
	}
	
	v, err := MatMul(value, mha.ValueWeight)
	if err != nil {
		return nil, fmt.Errorf("failed to project value: %v", err)
	}
	
	// In a full implementation, we would reshape to separate the heads here
	// For simplicity in this implementation, we're treating all heads as one
	
	// Compute attention scores
	kT, err := Transpose(k)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key: %v", err)
	}
	
	scores, err := MatMul(q, kT)
	if err != nil {
		return nil, fmt.Errorf("failed to compute attention scores: %v", err)
	}
	
	// Scale scores
	scaleFactor := math.Sqrt(float64(mha.HeadDim))
	scaledScores, err := ScalarMultiply(scores, 1.0/scaleFactor)
	if err != nil {
		return nil, fmt.Errorf("failed to scale attention scores: %v", err)
	}
	
	// Apply mask if provided and if using masked attention
	if mask != nil && mha.AttentionType == MaskedAttention {
		maskedScores, err := mask.ApplyMask(scaledScores)
		if err != nil {
			return nil, fmt.Errorf("failed to apply attention mask: %v", err)
		}
		scaledScores = maskedScores
	}
	
	// Apply softmax
	attentionWeights, err := Softmax(scaledScores)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax: %v", err)
	}
	
	// Apply dropout to attention weights if training and dropout is configured
	if isTraining && mha.DropoutRate > 0 {
		attentionWeights = mha.AttentionDropout.Forward(attentionWeights, true)
		if attentionWeights == nil {
			return nil, fmt.Errorf("dropout operation failed")
		}
	}
	
	// Apply attention weights to values
	weightedValues, err := MatMul(attentionWeights, v)
	if err != nil {
		return nil, fmt.Errorf("failed to apply attention weights: %v", err)
	}
	
	// Project back to model dimension
	output, err := MatMul(weightedValues, mha.OutputWeight)
	if err != nil {
		return nil, fmt.Errorf("failed to project output: %v", err)
	}
	
	return output, nil
}

// ForwardLegacy provides backward compatibility with the original API
// that doesn't require error handling
func (mha *MultiHeadAttention) ForwardLegacy(query, key, value *Matrix) *Matrix {
	output, err := mha.Forward(query, key, value, nil, false)
	if err != nil {
		// In legacy mode, we'll return a zero matrix on error
		zeroMatrix, _ := NewMatrix(query.Rows, mha.ModelDim)
		return zeroMatrix
	}
	return output
}

// Clone creates a deep copy of the MultiHeadAttention
func (mha *MultiHeadAttention) Clone() (*MultiHeadAttention, error) {
	queryWeight, err := mha.QueryWeight.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone query weight: %v", err)
	}
	
	keyWeight, err := mha.KeyWeight.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone key weight: %v", err)
	}
	
	valueWeight, err := mha.ValueWeight.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone value weight: %v", err)
	}
	
	outputWeight, err := mha.OutputWeight.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone output weight: %v", err)
	}
	
	return &MultiHeadAttention{
		NumHeads:        mha.NumHeads,
		ModelDim:        mha.ModelDim,
		HeadDim:         mha.HeadDim,
		QueryWeight:     queryWeight,
		KeyWeight:       keyWeight,
		ValueWeight:     valueWeight,
		OutputWeight:    outputWeight,
		AttentionDropout: mha.AttentionDropout.Clone(),
		DropoutRate:     mha.DropoutRate,
		AttentionType:   mha.AttentionType,
	}, nil
}

// Legacy compatibility functions to support older code
// These should be deprecated in future versions

// LegacyNewMultiHeadAttention creates a new multi-head attention layer without error checking
func LegacyNewMultiHeadAttention(numHeads, modelDim int) *MultiHeadAttention {
	mha, _ := NewStandardMultiHeadAttention(numHeads, modelDim)
	return mha
}

// LegacyNewAdvancedAttention creates a new advanced attention layer without error checking
func LegacyNewAdvancedAttention(numHeads, modelDim int, dropoutRate float64) *MultiHeadAttention {
	mha, _ := NewAdvancedMultiHeadAttention(numHeads, modelDim, dropoutRate)
	return mha
}
