package transformer

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
	EmbeddingMatrix   *Matrix
	OutputMatrix      *Matrix
	EmbeddingDropout  *Dropout
}

// EnhancedEncoderLayer represents an enhanced encoder layer with dropout
type EnhancedEncoderLayer struct {
	SelfAttention     *MultiHeadAttention
	FeedForward       *FeedForward
	Norm1             *LayerNorm
	Norm2             *LayerNorm
	DropoutAttention  *Dropout
	DropoutFFN        *Dropout
	DropoutResidual   *Dropout
	ModelDim          int
}

// NewEnhancedEncoderLayer creates a new enhanced encoder layer
func NewEnhancedEncoderLayer(modelDim, ffnHiddenDim, numHeads int, dropoutRate float64) *EnhancedEncoderLayer {
	return &EnhancedEncoderLayer{
		SelfAttention:    NewMultiHeadAttention(numHeads, modelDim),
		FeedForward:      NewFeedForward(modelDim, ffnHiddenDim),
		Norm1:            NewLayerNorm(modelDim),
		Norm2:            NewLayerNorm(modelDim),
		DropoutAttention: NewDropout(dropoutRate),
		DropoutFFN:       NewDropout(dropoutRate),
		DropoutResidual:  NewDropout(dropoutRate),
		ModelDim:         modelDim,
	}
}

// Forward processes input through the enhanced encoder layer
func (el *EnhancedEncoderLayer) Forward(x *Matrix, mask *AttentionMask, isTraining bool) *Matrix {
	// Self-attention with residual connection and layer norm
	attnOut := el.SelfAttention.Forward(x, x, x)
	
	// Apply dropout to attention output if training
	if isTraining {
		attnOut = el.DropoutAttention.Forward(attnOut, isTraining)
	}
	
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
	ffnOut := el.FeedForward.Forward(normalized1)
	
	// Apply dropout to FFN output if training
	if isTraining {
		ffnOut = el.DropoutFFN.Forward(ffnOut, isTraining)
	}
	
	// Add residual connection
	residual2 := NewMatrix(normalized1.Rows, normalized1.Cols)
	for i := 0; i < normalized1.Rows; i++ {
		for j := 0; j < normalized1.Cols; j++ {
			residual2.Data[i][j] = normalized1.Data[i][j] + ffnOut.Data[i][j]
		}
	}
	
	// Apply dropout to residual if training
	if isTraining {
		residual2 = el.DropoutResidual.Forward(residual2, isTraining)
	}
	
	// Apply layer normalization
	return el.Norm2.Forward(residual2)
}

// EnhancedDecoderLayer represents an enhanced decoder layer with dropout
type EnhancedDecoderLayer struct {
	SelfAttention     *MultiHeadAttention
	CrossAttention    *MultiHeadAttention
	FeedForward       *FeedForward
	Norm1             *LayerNorm
	Norm2             *LayerNorm
	Norm3             *LayerNorm
	DropoutSelfAttn   *Dropout
	DropoutCrossAttn  *Dropout
	DropoutFFN        *Dropout
	DropoutResidual1  *Dropout
	DropoutResidual2  *Dropout
	ModelDim          int
}

// NewEnhancedDecoderLayer creates a new enhanced decoder layer
func NewEnhancedDecoderLayer(modelDim, ffnHiddenDim, numHeads int, dropoutRate float64) *EnhancedDecoderLayer {
	return &EnhancedDecoderLayer{
		SelfAttention:    NewMultiHeadAttention(numHeads, modelDim),
		CrossAttention:   NewMultiHeadAttention(numHeads, modelDim),
		FeedForward:      NewFeedForward(modelDim, ffnHiddenDim),
		Norm1:            NewLayerNorm(modelDim),
		Norm2:            NewLayerNorm(modelDim),
		Norm3:            NewLayerNorm(modelDim),
		DropoutSelfAttn:  NewDropout(dropoutRate),
		DropoutCrossAttn: NewDropout(dropoutRate),
		DropoutFFN:       NewDropout(dropoutRate),
		DropoutResidual1: NewDropout(dropoutRate),
		DropoutResidual2: NewDropout(dropoutRate),
		ModelDim:         modelDim,
	}
}

// Forward processes input through the enhanced decoder layer
func (dl *EnhancedDecoderLayer) Forward(x, encoderOutput *Matrix, selfMask, crossMask *AttentionMask, isTraining bool) *Matrix {
	// Self-attention with residual connection and layer norm
	selfAttnOut := dl.SelfAttention.Forward(x, x, x)
	
	// Apply dropout to self-attention output if training
	if isTraining {
		selfAttnOut = dl.DropoutSelfAttn.Forward(selfAttnOut, isTraining)
	}
	
	// Add residual connection
	residual1 := NewMatrix(x.Rows, x.Cols)
	for i := 0; i < x.Rows; i++ {
		for j := 0; j < x.Cols; j++ {
			residual1.Data[i][j] = x.Data[i][j] + selfAttnOut.Data[i][j]
		}
	}
	
	// Apply dropout to residual if training
	if isTraining {
		residual1 = dl.DropoutResidual1.Forward(residual1, isTraining)
	}
	
	// Apply layer normalization
	normalized1 := dl.Norm1.Forward(residual1)
	
	// Cross-attention with encoder output
	crossAttnOut := dl.CrossAttention.Forward(normalized1, encoderOutput, encoderOutput)
	
	// Apply dropout to cross-attention output if training
	if isTraining {
		crossAttnOut = dl.DropoutCrossAttn.Forward(crossAttnOut, isTraining)
	}
	
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
	ffnOut := dl.FeedForward.Forward(normalized2)
	
	// Apply dropout to FFN output if training
	if isTraining {
		ffnOut = dl.DropoutFFN.Forward(ffnOut, isTraining)
	}
	
	// Add residual connection
	residual3 := NewMatrix(normalized2.Rows, normalized2.Cols)
	for i := 0; i < normalized2.Rows; i++ {
		for j := 0; j < normalized2.Cols; j++ {
			residual3.Data[i][j] = normalized2.Data[i][j] + ffnOut.Data[i][j]
		}
	}
	
	// Apply dropout to residual if training
	if isTraining {
		residual3 = dl.DropoutResidual2.Forward(residual3, isTraining)
	}
	
	// Apply layer normalization
	return dl.Norm3.Forward(residual3)
}
