package core

import (
	"transformer/internal/utils" // Added utils import
	"fmt" // For errors
)

// EncoderLayer represents a single encoder layer in a transformer
type EncoderLayer struct {
	SelfAttention *MultiHeadAttention // core.MultiHeadAttention
	FeedForward   *FeedForward        // core.FeedForward
	Norm1         *LayerNorm          // core.LayerNorm
	Norm2         *LayerNorm          // core.LayerNorm
	ModelDim      int
}

// NewEncoderLayer creates a new encoder layer
// This function should ideally return an error if component constructors fail
func NewEncoderLayer(modelDim, ffnHiddenDim, numHeads int) *EncoderLayer {
	// NewDefaultFeedForward now returns (*FeedForward, error)
	ff, errFf := NewDefaultFeedForward(modelDim, ffnHiddenDim)
	if errFf != nil {
		panic(fmt.Sprintf("failed to create FeedForward in NewEncoderLayer: %v", errFf)) // Or return error
	}
	return &EncoderLayer{
		SelfAttention: NewMultiHeadAttention(numHeads, modelDim), // core.NewMultiHeadAttention
		FeedForward:   ff,
		Norm1:         NewLayerNorm(modelDim), // core.NewLayerNorm
		Norm2:         NewLayerNorm(modelDim), // core.NewLayerNorm
		ModelDim:      modelDim,
	}
}

// Forward processes input through the encoder layer
func (el *EncoderLayer) Forward(x *utils.Matrix) *utils.Matrix { // Prefixed Matrix
	// Self-attention with residual connection and layer norm
	// el.SelfAttention.Forward, el.Norm1.Forward etc. are placeholder methods for now
	// and don't return errors. If they did, this Forward method would also need to return an error.
	attnOut := el.SelfAttention.Forward(x, x, x)
	
	// Add residual connection
	residual1, _ := utils.NewMatrix(x.Rows, x.Cols) // Prefixed, ignoring error for deprecated code
	if residual1 != nil { // Basic check
		for i := 0; i < x.Rows; i++ {
			for j := 0; j < x.Cols; j++ {
				residual1.Data[i][j] = x.Data[i][j] + attnOut.Data[i][j]
			}
		}
	} else {
		// This path should not be reached if NewMatrix panics on error or error is handled.
		// If NewMatrix can return nil without error, this is problematic.
		// For now, assume error handling in NewMatrix would panic or be caught earlier.
		return nil // Or some error indication
	}
	
	// Apply layer normalization
	normalized1 := el.Norm1.Forward(residual1)
	
	// Feed-forward network
	ffnOut := el.FeedForward.ForwardLegacy(normalized1)
	
	// Add residual connection
	residual2, _ := utils.NewMatrix(normalized1.Rows, normalized1.Cols) // Prefixed, ignoring error
	if residual2 != nil {
		for i := 0; i < normalized1.Rows; i++ {
			for j := 0; j < normalized1.Cols; j++ {
				residual2.Data[i][j] = normalized1.Data[i][j] + ffnOut.Data[i][j]
			}
		}
	} else {
		return nil
	}
	
	// Apply layer normalization
	return el.Norm2.Forward(residual2)
}

// mustFeedForward is a helper function to panic on error
// This function is no longer used if NewEncoderLayer handles the error from NewDefaultFeedForward
/*
func mustFeedForward(ff *FeedForward, err error) *FeedForward {
	if err != nil {
		panic(err)
	}
	return ff
}
*/
// Removed extra closing brace
