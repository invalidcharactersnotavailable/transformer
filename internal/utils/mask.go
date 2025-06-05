package transformer

import (
	"transformer/pkg/autodiff"
)

// AttentionMask represents a mask for controlling attention flow
type AttentionMask struct {
	Mask *autodiff.Matrix
}

// NewPaddingMask creates a mask for padding tokens
func NewPaddingMask(seqLen int, validLengths []int) *AttentionMask {
	mask := autodiff.MustNewMatrix(len(validLengths), seqLen)
	
	// Set 1.0 for valid positions, 0.0 for padding
	for i, validLen := range validLengths {
		for j := 0; j < seqLen; j++ {
			if j < validLen {
				mask.Data[i][j] = 1.0
			} else {
				mask.Data[i][j] = 0.0
			}
		}
	}
	
	return &AttentionMask{Mask: mask}
}

// NewCausalMask creates a causal (future-blinding) mask for decoder
func NewCausalMask(seqLen int) *AttentionMask {
	mask := autodiff.MustNewMatrix(seqLen, seqLen)
	
	// Set 1.0 for positions that can be attended to (lower triangle)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				mask.Data[i][j] = 1.0
			} else {
				mask.Data[i][j] = 0.0
			}
		}
	}
	
	return &AttentionMask{Mask: mask}
}

// ApplyMask applies the attention mask to attention scores
func (am *AttentionMask) ApplyMask(scores *autodiff.Matrix) *autodiff.Matrix {
	result := autodiff.MustNewMatrix(scores.Rows, scores.Cols)
	
	for i := 0; i < scores.Rows; i++ {
		for j := 0; j < scores.Cols; j++ {
			maskValue := am.Mask.Data[i % am.Mask.Rows][j % am.Mask.Cols]
			if maskValue > 0.0 {
				result.Data[i][j] = scores.Data[i][j]
			} else {
				// Set to negative infinity (represented by a very large negative number)
				result.Data[i][j] = -1e9
			}
		}
	}
	
	return result
}
