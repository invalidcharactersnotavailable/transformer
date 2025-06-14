package transformer

import (
	"fmt"
	"math"
)

// Helper functions for matrix operations

// sliceCols extracts a new matrix by slicing specified columns from an existing matrix.
func sliceCols(m *Matrix, startCol, endCol int) (*Matrix, error) {
	if m == nil {
		return nil, fmt.Errorf("input matrix cannot be nil")
	}
	if startCol < 0 || endCol <= startCol || endCol > m.Cols {
		return nil, fmt.Errorf("invalid column slice indices: start %d, end %d for matrix with %d cols", startCol, endCol, m.Cols)
	}
	numSlicedCols := endCol - startCol
	sliced, err := NewMatrix(m.Rows, numSlicedCols)
	if err != nil {
		return nil, fmt.Errorf("failed to create new matrix for slice: %v", err)
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < numSlicedCols; j++ {
			sliced.Data[i][j] = m.Data[i][startCol+j]
		}
	}
	return sliced, nil
}

// concatenateCols concatenates a list of matrices column-wise.
// All matrices must have the same number of rows.
func concatenateCols(matrices []*Matrix) (*Matrix, error) {
	if len(matrices) == 0 {
		return nil, fmt.Errorf("cannot concatenate empty list of matrices")
	}
	numRows := matrices[0].Rows
	totalCols := 0
	for _, m := range matrices {
		if m.Rows != numRows {
			return nil, fmt.Errorf("matrices must have the same number of rows to concatenate column-wise")
		}
		totalCols += m.Cols
	}

	concatenated, err := NewMatrix(numRows, totalCols)
	if err != nil {
		return nil, fmt.Errorf("failed to create new matrix for concatenation: %v", err)
	}

	currentStartCol := 0
	for _, m := range matrices {
		for i := 0; i < m.Rows; i++ {
			for j := 0; j < m.Cols; j++ {
				concatenated.Data[i][currentStartCol+j] = m.Data[i][j]
			}
		}
		currentStartCol += m.Cols
	}
	return concatenated, nil
}

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
	seqLen := query.Rows // Assuming query, key, value have same seq_len (num rows)

	// 1. Initial linear projections
	qFull, err := MatMul(query, mha.QueryWeight)
	if err != nil { return nil, fmt.Errorf("query projection failed: %v", err) }
	kFull, err := MatMul(key, mha.KeyWeight)
	if err != nil { return nil, fmt.Errorf("key projection failed: %v", err) }
	vFull, err := MatMul(value, mha.ValueWeight)
	if err != nil { return nil, fmt.Errorf("value projection failed: %v", err) }

	headOutputs := make([]*Matrix, 0, mha.NumHeads)

	// 2. Process each head
	for h := 0; h < mha.NumHeads; h++ {
		startCol := h * mha.HeadDim
		endCol := (h + 1) * mha.HeadDim

		qHead, err := sliceCols(qFull, startCol, endCol)
		if err != nil { return nil, fmt.Errorf("slicing qFull for head %d failed: %v", h, err) }
		kHead, err := sliceCols(kFull, startCol, endCol)
		if err != nil { return nil, fmt.Errorf("slicing kFull for head %d failed: %v", h, err) }
		vHead, err := sliceCols(vFull, startCol, endCol)
		if err != nil { return nil, fmt.Errorf("slicing vFull for head %d failed: %v", h, err) }

		// Scaled Dot-Product Attention for the current head
		kHeadT, err := Transpose(kHead)
		if err != nil { return nil, fmt.Errorf("transpose kHead for head %d failed: %v", h, err) }

		scores, err := MatMul(qHead, kHeadT)
		if err != nil { return nil, fmt.Errorf("matmul qHead*kHeadT for head %d failed: %v", h, err) }

		scaledScores, err := ScalarMultiply(scores, 1.0/math.Sqrt(float64(mha.HeadDim)))
		if err != nil { return nil, fmt.Errorf("scaling scores for head %d failed: %v", h, err) }

		if mask != nil && mha.AttentionType == MaskedAttention {
			// Ensure mask is (seqLen, seqLen)
			if mask.Mask.Rows != seqLen || mask.Mask.Cols != seqLen {
				return nil, fmt.Errorf("attention mask dimensions (%dx%d) incompatible with sequence length %d for head %d", mask.Mask.Rows, mask.Mask.Cols, seqLen, h)
			}
			scaledScores, err = mask.ApplyMask(scaledScores)
			if err != nil { return nil, fmt.Errorf("applying mask for head %d failed: %v", h, err) }
		}

		attentionWeights, err := Softmax(scaledScores)
		if err != nil { return nil, fmt.Errorf("softmax for head %d failed: %v", h, err) }

		if isTraining && mha.DropoutRate > 0 {
			attentionWeights = mha.AttentionDropout.Forward(attentionWeights, true) // Forward method of Dropout doesn't return error
			if attentionWeights == nil { // Check if dropout failed (though current dropout doesn't indicate failure this way)
				return nil, fmt.Errorf("dropout on attention weights failed for head %d", h)
			}
		}

		weightedValue, err := MatMul(attentionWeights, vHead)
		if err != nil { return nil, fmt.Errorf("matmul attentionWeights*vHead for head %d failed: %v", h, err) }

		headOutputs = append(headOutputs, weightedValue)
	}

	// 3. Concatenate head outputs
	var concatenatedOutput *Matrix
	if len(headOutputs) == 1 { // Optimization for single head
		 concatenatedOutput = headOutputs[0]
	} else {
		 concatenatedOutput, err = concatenateCols(headOutputs)
		 if err != nil { return nil, fmt.Errorf("concatenating head outputs failed: %v", err) }
	}
	// Check if concatenatedOutput has the expected model_dim columns
	if concatenatedOutput.Cols != mha.ModelDim {
		 // This might happen if HeadDim * NumHeads != ModelDim, but constructor already checks ModelDim % NumHeads == 0.
		 // More likely an issue if concatenateCols has a bug or if headOutputs is empty.
		 return nil, fmt.Errorf("concatenated output has %d columns, expected model_dim %d", concatenatedOutput.Cols, mha.ModelDim)
	}


	// 4. Final linear projection
	finalOutput, err := MatMul(concatenatedOutput, mha.OutputWeight)
	if err != nil { return nil, fmt.Errorf("final output projection failed: %v", err) }

	return finalOutput, nil
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
