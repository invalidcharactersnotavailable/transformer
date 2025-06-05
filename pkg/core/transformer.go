package core

import (
	"transformer/internal/utils" // For utils.Matrix, utils.NewMatrix etc.
	"fmt" // For errors
	// "math" // Removed unused import
)

// --- Restored/Added Matrix-based component definitions ---

// MultiHeadAttention defines a matrix-based multi-head attention layer.
type MultiHeadAttention struct {
	NumHeads     int
	ModelDim     int
	HeadDim      int
	QueryWeight  *utils.Matrix
	KeyWeight    *utils.Matrix
	ValueWeight  *utils.Matrix
	OutputWeight *utils.Matrix
	// Add other fields like dropout if necessary based on utils.Dropout
}

// NewMultiHeadAttention creates a new matrix-based MultiHeadAttention layer.
func NewMultiHeadAttention(numHeads, modelDim int) *MultiHeadAttention {
	if modelDim%numHeads != 0 {
		panic(fmt.Sprintf("modelDim (%d) must be divisible by numHeads (%d)", modelDim, numHeads))
	}
	headDim := modelDim / numHeads
	return &MultiHeadAttention{
		NumHeads:     numHeads,
		ModelDim:     modelDim,
		HeadDim:      headDim,
		QueryWeight:  utils.MustNewRandomMatrix(modelDim, modelDim), // Using Must for simplicity here
		KeyWeight:    utils.MustNewRandomMatrix(modelDim, modelDim),
		ValueWeight:  utils.MustNewRandomMatrix(modelDim, modelDim),
		OutputWeight: utils.MustNewRandomMatrix(modelDim, modelDim),
	}
}

// Forward method for matrix-based MultiHeadAttention (simplified placeholder)
func (mha *MultiHeadAttention) Forward(query, key, value *utils.Matrix) *utils.Matrix {
	// Placeholder: In a real implementation, this would perform attention logic
	// For now, just return a matrix of the expected output shape.
	// Output of MHA is typically (seq_len, model_dim) which is same as query shape.
	out, _ := utils.NewMatrix(query.Rows, query.Cols) // Error handling omitted for placeholder
	return out
}

// FeedForward defines a matrix-based feed-forward layer.
type FeedForward struct {
	W1 *utils.Matrix
	B1 *utils.Matrix
	W2 *utils.Matrix
	B2 *utils.Matrix
	// Add activation function if needed
}

// NewDefaultFeedForward creates a new matrix-based FeedForward layer with default sizes.
// Note: The original call was NewDefaultFeedForward(modelDim, ffnHiddenDim)
// The mustFeedForward wrapper implies NewDefaultFeedForward might return an error.
func NewDefaultFeedForward(inputDim, hiddenDim int) (*FeedForward, error) {
	w1, err := utils.NewRandomMatrix(inputDim, hiddenDim)
	if err != nil { return nil, err }
	b1, err := utils.NewMatrix(1, hiddenDim) // Changed from NewZerosMatrix
	if err != nil { return nil, err }
	w2, err := utils.NewRandomMatrix(hiddenDim, inputDim)
	if err != nil { return nil, err }
	b2, err := utils.NewMatrix(1, inputDim) // Changed from NewZerosMatrix
	if err != nil { return nil, err }

	return &FeedForward{W1: w1, B1: b1, W2: w2, B2: b2}, nil
}

// ForwardLegacy for matrix-based FeedForward (simplified placeholder)
// Original call in encoder/decoder was ForwardLegacy
func (ff *FeedForward) ForwardLegacy(input *utils.Matrix) *utils.Matrix {
	// Placeholder
	out, _ := utils.NewMatrix(input.Rows, ff.W2.Cols) // W2.Cols is outputDim
	return out
}


// LayerNorm defines a matrix-based layer normalization.
type LayerNorm struct {
	Dim     int
	Gamma   *utils.Matrix
	Beta    *utils.Matrix
	Epsilon float64
}

// NewLayerNorm creates a new matrix-based LayerNorm.
// This function should probably return an error as utils.NewOnesMatrix and utils.NewMatrix do.
func NewLayerNorm(dim int) *LayerNorm {
	gamma, errGamma := utils.NewOnesMatrix(1, dim)
	beta, errBeta := utils.NewMatrix(1, dim)
	if errGamma != nil || errBeta != nil {
		// In a real scenario, propagate these errors. Panicking for deprecated code.
		panic(fmt.Sprintf("Error creating LayerNorm matrices: gammaErr=%v, betaErr=%v", errGamma, errBeta))
	}
	return &LayerNorm{
		Dim:     dim,
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: 1e-5, // Default epsilon
	}
}

// Forward method for matrix-based LayerNorm (simplified placeholder)
func (ln *LayerNorm) Forward(input *utils.Matrix) *utils.Matrix {
	// Placeholder
	out, _ := utils.NewMatrix(input.Rows, input.Cols)
	return out
}


// DEPRECATED: This Transformer uses Matrix-based parameters and is not connected to
// the automatic differentiation system. It is not suitable for training with the
// current gradient-based fine-tuning pipeline.
// For a trainable model, please use `TransformerWithTensors` from the `autodiff` package
// and the `GradientFineTuner` from `autodiff` package.
type Transformer struct {
	Encoder           []*EncoderLayer       // Defined in encoder.go
	Decoder           []*DecoderLayer       // Defined in decoder.go
	PositionalEncoder *PositionalEncoding // Defined in merged_positional.go
	EmbeddingDim      int
	VocabSize         int
	EmbeddingMatrix   *utils.Matrix // Prefixed
	OutputMatrix      *utils.Matrix // Prefixed
}

// NewTransformer creates a new transformer model
// This function will need to handle errors from NewEncoderLayer, NewDecoderLayer, NewPositionalEncoding, NewRandomMatrix
func NewTransformer(vocabSize, embeddingDim, numLayers, numHeads, ffnHiddenDim, maxLen int) *Transformer {
	// Create encoder layers
	encoder := make([]*EncoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		// NewEncoderLayer will need to be updated to handle errors from its own component constructors
		// or this call needs to handle an error from NewEncoderLayer if its signature changes.
		encoder[i] = NewEncoderLayer(embeddingDim, ffnHiddenDim, numHeads)
	}
	
	// Create decoder layers
	decoder := make([]*DecoderLayer, numLayers)
	for i := 0; i < numLayers; i++ {
		// Similarly, NewDecoderLayer might need error handling updates.
		decoder[i] = NewDecoderLayer(embeddingDim, ffnHiddenDim, numHeads)
	}
	
	// NewPositionalEncoding now returns (*PositionalEncoding, error)
	posEnc, _ := NewPositionalEncoding(&PositionalEncodingConfig{Dim: embeddingDim, MaxLen: maxLen}) // Simplified error handling

	// utils.NewRandomMatrix returns (*utils.Matrix, error)
	embMatrix, _ := utils.NewRandomMatrix(vocabSize, embeddingDim) // Simplified error handling
	outMatrix, _ := utils.NewRandomMatrix(embeddingDim, vocabSize) // Simplified error handling

	return &Transformer{
		Encoder:           encoder,
		Decoder:           decoder,
		PositionalEncoder: posEnc,
		EmbeddingDim:      embeddingDim,
		VocabSize:         vocabSize,
		EmbeddingMatrix:   embMatrix,
		OutputMatrix:      outMatrix,
	}
}

// Embed converts token indices to embeddings
func (t *Transformer) Embed(indices []int) *utils.Matrix { // Prefixed
	// utils.NewMatrix returns (*utils.Matrix, error)
	result, _ := utils.NewMatrix(len(indices), t.EmbeddingDim) // Simplified error handling
	
	if result == nil { // Basic check if NewMatrix failed (though error is ignored above)
		return nil // Or handle error more robustly
	}

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
func (t *Transformer) Forward(srcIndices, tgtIndices []int) *utils.Matrix { // Prefixed
	srcEmbeddings := t.Embed(srcIndices)
	tgtEmbeddings := t.Embed(tgtIndices)
	
	// t.PositionalEncoder.AddToEmbedding now returns (*utils.Matrix, error)
	srcEmbeddingsWithPos, _ := t.PositionalEncoder.AddToEmbedding(srcEmbeddings, false) // false for isTraining, simplified error handling
	tgtEmbeddingsWithPos, _ := t.PositionalEncoder.AddToEmbedding(tgtEmbeddings, false) // Simplified error handling
	
	encoderOutput := srcEmbeddingsWithPos
	for _, layer := range t.Encoder {
		// layer.Forward (EncoderLayer.Forward) will need to return error
		encoderOutput = layer.Forward(encoderOutput) // Simplified, assumes no error or panics
	}
	
	decoderOutput := tgtEmbeddingsWithPos
	for _, layer := range t.Decoder {
		// layer.Forward (DecoderLayer.Forward) will need to return error
		decoderOutput = layer.Forward(decoderOutput, encoderOutput) // Simplified
	}
	
	// utils.MatMul and utils.Softmax return (*utils.Matrix, error)
	logits, _ := utils.MatMul(decoderOutput, t.OutputMatrix) // Simplified
	probs, _ := utils.Softmax(logits)                        // Simplified
	
	return probs
}
