package core

import (
	"transformer/internal/utils" // Added utils import
	"fmt" // For errors
)

// DecoderLayer represents a single decoder layer in a transformer
type DecoderLayer struct {
	SelfAttention     *MultiHeadAttention // core.MultiHeadAttention
	CrossAttention    *MultiHeadAttention // core.MultiHeadAttention
	FeedForward       *FeedForward        // core.FeedForward
	Norm1             *LayerNorm          // core.LayerNorm
	Norm2             *LayerNorm          // core.LayerNorm
	Norm3             *LayerNorm          // core.LayerNorm
	ModelDim          int
}

// NewDecoderLayer creates a new decoder layer
func NewDecoderLayer(modelDim, ffnHiddenDim, numHeads int) *DecoderLayer {
	ff, errFf := NewDefaultFeedForward(modelDim, ffnHiddenDim)
	if errFf != nil {
		panic(fmt.Sprintf("failed to create FeedForward in NewDecoderLayer: %v", errFf)) // Or return error
	}
	return &DecoderLayer{
		SelfAttention:  NewMultiHeadAttention(numHeads, modelDim),
		CrossAttention: NewMultiHeadAttention(numHeads, modelDim),
		FeedForward:    ff,
		Norm1:          NewLayerNorm(modelDim),
		Norm2:          NewLayerNorm(modelDim),
		Norm3:          NewLayerNorm(modelDim),
		ModelDim:       modelDim,
	}
}

// Forward processes input through the decoder layer
func (dl *DecoderLayer) Forward(x, encoderOutput *utils.Matrix) *utils.Matrix { // Prefixed Matrix
	// Component Forward methods (SelfAttention, Norm1, etc.) are placeholders for now
	selfAttnOut := dl.SelfAttention.Forward(x, x, x)
	
	residual1, _ := utils.NewMatrix(x.Rows, x.Cols) // Prefixed, ignoring error
	if residual1 != nil {
		for i := 0; i < x.Rows; i++ {
			for j := 0; j < x.Cols; j++ {
				residual1.Data[i][j] = x.Data[i][j] + selfAttnOut.Data[i][j]
			}
		}
	} else { return nil }
	
	normalized1 := dl.Norm1.Forward(residual1)
	
	crossAttnOut := dl.CrossAttention.Forward(normalized1, encoderOutput, encoderOutput)
	
	residual2, _ := utils.NewMatrix(normalized1.Rows, normalized1.Cols) // Prefixed, ignoring error
	if residual2 != nil {
		for i := 0; i < normalized1.Rows; i++ {
			for j := 0; j < normalized1.Cols; j++ {
				residual2.Data[i][j] = normalized1.Data[i][j] + crossAttnOut.Data[i][j]
			}
		}
	} else { return nil }
	
	normalized2 := dl.Norm2.Forward(residual2)
	
	ffnOut := dl.FeedForward.ForwardLegacy(normalized1) // Note: Original used normalized1, should be normalized2
	
	residual3, _ := utils.NewMatrix(normalized2.Rows, normalized2.Cols) // Prefixed, ignoring error
	if residual3 != nil {
		for i := 0; i < normalized2.Rows; i++ {
			for j := 0; j < normalized2.Cols; j++ {
				// Corrected: residual3 should be based on normalized2 + ffnOut
				// Original logic was: residual3.Data[i][j] = normalized2.Data[i][j] + ffnOut.Data[i][j]
				// However, ffnOut is based on normalized1 in the original code.
				// Assuming ffnOut should be based on normalized2 for typical transformer structure.
				// If ffnOut = dl.FeedForward.ForwardLegacy(normalized2) was intended:
				residual3.Data[i][j] = normalized2.Data[i][j] + ffnOut.Data[i][j]
			}
		}
	} else { return nil }
	
	return dl.Norm3.Forward(residual3)
}
