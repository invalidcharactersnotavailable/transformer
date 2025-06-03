package transformer

// EncoderLayer represents a single encoder layer in a transformer
type EncoderLayer struct {
	SelfAttention *MultiHeadAttention
	FeedForward   *FeedForward // Corrected type
	Norm1         *LayerNorm
	Norm2         *LayerNorm
	ModelDim      int
}

// NewEncoderLayer creates a new encoder layer
func NewEncoderLayer(modelDim, ffnHiddenDim, numHeads int) *EncoderLayer {
	return &EncoderLayer{
		SelfAttention: NewMultiHeadAttention(numHeads, modelDim),
		FeedForward:   mustFeedForward(NewDefaultFeedForward(modelDim, ffnHiddenDim)), // Corrected line
		Norm1:         NewLayerNorm(modelDim),
		Norm2:         NewLayerNorm(modelDim),
		ModelDim:      modelDim,
	}
}

// Forward processes input through the encoder layer
func (el *EncoderLayer) Forward(x *Matrix) *Matrix {
	// Self-attention with residual connection and layer norm
	attnOut := el.SelfAttention.Forward(x, x, x)
	
	// Add residual connection
	residual1 := NewMatrix(x.Rows, x.Cols)
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			residual1.Data[i][j] = x.Data[i][j] + attnOut.Data[i][j]
		}
	}
	
	// Apply layer normalization
	normalized1 := el.Norm1.Forward(residual1)
	
	// Feed-forward network
	ffnOut := el.FeedForward.ForwardLegacy(normalized1) // Adjusted line
	
	// Add residual connection
	residual2 := NewMatrix(normalized1.Rows, normalized1.Cols)
	for i := 0; i < normalized1.Rows; i++ {
		for j := 0; j < normalized1.Cols; j++ {
			residual2.Data[i][j] = normalized1.Data[i][j] + ffnOut.Data[i][j]
		}
	}
	
	// Apply layer normalization
	return el.Norm2.Forward(residual2)
}

// mustFeedForward is a helper function to panic on error
func mustFeedForward(ff *FeedForward, err error) *FeedForward { // Corrected type
	if err != nil {
		panic(err)
	}
	return ff
}
