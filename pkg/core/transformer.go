package transformer

// DEPRECATED: This Transformer uses Matrix-based parameters and is not connected to
// the automatic differentiation system. It is not suitable for training with the
// current gradient-based fine-tuning pipeline.
// For a trainable model, please use `TransformerWithTensors` from the `autodiff` package
// and the `GradientFineTuner` from `autodiff` package.
type Transformer struct {
	Encoder           []*EncoderLayer
	Decoder           []*DecoderLayer
	PositionalEncoder *PositionalEncoding
	EmbeddingDim      int
	VocabSize         int
	EmbeddingMatrix   *Matrix
	OutputMatrix      *Matrix
}

// NewTransformer creates a new transformer model
func NewTransformer(vocabSize, embeddingDim, numLayers, numHeads, ffnHiddenDim, maxLen int) *Transformer {
	// Create encoder layers
	encoder := make([]*EncoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		encoder[i] = NewEncoderLayer(embeddingDim, ffnHiddenDim, numHeads)
	}
	
	// Create decoder layers
	decoder := make([]*DecoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		decoder[i] = NewDecoderLayer(embeddingDim, ffnHiddenDim, numHeads)
	}
	
	return &Transformer{
		Encoder:           encoder,
		Decoder:           decoder,
		PositionalEncoder: NewPositionalEncoding(embeddingDim, maxLen),
		EmbeddingDim:      embeddingDim,
		VocabSize:         vocabSize,
		EmbeddingMatrix:   NewRandomMatrix(vocabSize, embeddingDim),
		OutputMatrix:      NewRandomMatrix(embeddingDim, vocabSize),
	}
}

// Embed converts token indices to embeddings
func (t *Transformer) Embed(indices []int) *Matrix {
	result := NewMatrix(len(indices), t.EmbeddingDim)
	
	for i, idx := range indices {
		if idx >= 0 && idx < t.VocabSize {
			for j := 0; j < t.EmbeddingDim; j++ {
				result.Data[i][j] = t.EmbeddingMatrix.Data[idx][j]
			}
		}
	}
	
	return result
}

// Forward processes input through the transformer model
func (t *Transformer) Forward(srcIndices, tgtIndices []int) *Matrix {
	// Convert token indices to embeddings
	srcEmbeddings := t.Embed(srcIndices)
	tgtEmbeddings := t.Embed(tgtIndices)
	
	// Add positional encoding
	srcEmbeddings = t.PositionalEncoder.AddToEmbedding(srcEmbeddings)
	tgtEmbeddings = t.PositionalEncoder.AddToEmbedding(tgtEmbeddings)
	
	// Process through encoder
	encoderOutput := srcEmbeddings
	for _, layer := range t.Encoder {
		encoderOutput = layer.Forward(encoderOutput)
	}
	
	// Process through decoder
	decoderOutput := tgtEmbeddings
	for _, layer := range t.Decoder {
		decoderOutput = layer.Forward(decoderOutput, encoderOutput)
	}
	
	// Project to vocabulary size
	logits := MatMul(decoderOutput, t.OutputMatrix)
	
	// Apply softmax to get probabilities
	return Softmax(logits)
}
