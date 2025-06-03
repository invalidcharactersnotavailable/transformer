package transformer

// DecoderLayer represents a single decoder layer in a transformer
type DecoderLayer struct {
	SelfAttention     *MultiHeadAttention
	CrossAttention    *MultiHeadAttention
	FeedForward       *FeedForward // Corrected type
	Norm1             *LayerNorm
	Norm2             *LayerNorm
	Norm3             *LayerNorm
	ModelDim          int
}

// NewDecoderLayer creates a new decoder layer
func NewDecoderLayer(modelDim, ffnHiddenDim, numHeads int) *DecoderLayer {
	return &DecoderLayer{
		SelfAttention:  NewMultiHeadAttention(numHeads, modelDim),
		CrossAttention: NewMultiHeadAttention(numHeads, modelDim),
		FeedForward:    mustFeedForward(NewDefaultFeedForward(modelDim, ffnHiddenDim)), // Corrected line
		Norm1:          NewLayerNorm(modelDim),
		Norm2:          NewLayerNorm(modelDim),
		Norm3:          NewLayerNorm(modelDim),
		ModelDim:       modelDim,
	}
}

// Forward processes input through the decoder layer
func (dl *DecoderLayer) Forward(x, encoderOutput *Matrix) *Matrix {
	// Self-attention with residual connection and layer norm
	selfAttnOut := dl.SelfAttention.Forward(x, x, x)
	
	// Add residual connection
	residual1 := NewMatrix(x.Rows, x.Cols)
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			residual1.Data[i][j] = x.Data[i][j] + selfAttnOut.Data[i][j]
		}
	}
	
	// Apply layer normalization
	normalized1 := dl.Norm1.Forward(residual1)
	
	// Cross-attention with encoder output
	crossAttnOut := dl.CrossAttention.Forward(normalized1, encoderOutput, encoderOutput)
	
	// Add residual connection
	residual2 := NewMatrix(normalized1.Rows, normalized1.Cols)
	for i := 0; i < normalized1.Rows; i++ {
		for j := 0; j < normalized1.Cols; j++ {
			residual2.Data[i][j] = normalized1.Data[i][j] + crossAttnOut.Data[i][j]
		}
	}
	
	// Apply layer normalization
	normalized2 := dl.Norm2.Forward(residual2)
	
	// Feed-forward network
	ffnOut := dl.FeedForward.ForwardLegacy(normalized2) // Adjusted line
	
	// Add residual connection
	residual3 := NewMatrix(normalized2.Rows, normalized2.Cols)
	for i := 0; i < normalized2.Rows; i++ {
		for j := 0; j < normalized2.Cols; j++ {
			residual3.Data[i][j] = normalized2.Data[i][j] + ffnOut.Data[i][j]
		}
	}
	
	// Apply layer normalization
	return dl.Norm3.Forward(residual3)
}
