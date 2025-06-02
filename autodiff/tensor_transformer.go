package autodiff

import (
	"fmt"
	"math"
)

// TransformerWithTensors represents the complete transformer model with tensor-based parameters
type TransformerWithTensors struct {
	Encoder           []*EncoderLayerWithTensors
	Decoder           []*DecoderLayerWithTensors
	PositionalEncoder *PositionalEncoding
	EmbeddingDim      int
	VocabSize         int
	EmbeddingMatrix   *Tensor
	OutputMatrix      *Tensor
}

// EncoderLayerWithTensors represents a single encoder layer with tensor-based parameters
type EncoderLayerWithTensors struct {
	SelfAttention    *MultiHeadAttentionWithTensors
	FeedForward      *FeedForwardWithTensors
	Norm1            *LayerNormWithTensors
	Norm2            *LayerNormWithTensors
	DropoutAttention *Dropout
	DropoutFFN       *Dropout
	DropoutResidual  *Dropout
}

// DecoderLayerWithTensors represents a single decoder layer with tensor-based parameters
type DecoderLayerWithTensors struct {
	SelfAttention     *MultiHeadAttentionWithTensors
	CrossAttention    *MultiHeadAttentionWithTensors
	FeedForward       *FeedForwardWithTensors
	Norm1             *LayerNormWithTensors
	Norm2             *LayerNormWithTensors
	Norm3             *LayerNormWithTensors
	DropoutSelfAttn   *Dropout
	DropoutCrossAttn  *Dropout
	DropoutFFN        *Dropout
	DropoutResidual1  *Dropout
	DropoutResidual2  *Dropout
}

// MultiHeadAttentionWithTensors represents a multi-head attention mechanism with tensor-based parameters
type MultiHeadAttentionWithTensors struct {
	NumHeads    int
	ModelDim    int
	HeadDim     int
	QueryWeight *Tensor
	KeyWeight   *Tensor
	ValueWeight *Tensor
	OutputWeight *Tensor
}

// FeedForwardWithTensors represents a feed-forward neural network with tensor-based parameters
type FeedForwardWithTensors struct {
	InputDim  int
	HiddenDim int
	W1        *Tensor
	B1        *Tensor
	W2        *Tensor
	B2        *Tensor
}

// LayerNormWithTensors represents layer normalization with tensor-based parameters
type LayerNormWithTensors struct {
	Dim    int
	Gamma  *Tensor
	Beta   *Tensor
	Eps    float64
}

// PositionalEncoding represents positional encoding for transformer inputs
type PositionalEncoding struct {
	Dim      int
	MaxLen   int
	Encoding *Matrix
}

// Dropout represents dropout for regularization
type Dropout struct {
	Rate float64
}

// NewDropout creates a new dropout layer
func NewDropout(rate float64) *Dropout {
	return &Dropout{Rate: rate}
}

// ShouldDrop returns whether to drop a value
func (d *Dropout) ShouldDrop() bool {
	return rand.Float64() < d.Rate
}

// NewPositionalEncoding creates a new positional encoding component
func NewPositionalEncoding(dim, maxLen int) *PositionalEncoding {
	encoding := NewMatrix(maxLen, dim)
	
	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dim; i += 2 {
			// Calculate the sine and cosine values
			denominator := math.Pow(10000, float64(i)/float64(dim))
			
			// Sine for even indices
			if i < dim {
				encoding.Data[pos][i] = math.Sin(float64(pos) / denominator)
			}
			
			// Cosine for odd indices
			if i+1 < dim {
				encoding.Data[pos][i+1] = math.Cos(float64(pos) / denominator)
			}
		}
	}
	
	return &PositionalEncoding{
		Dim:      dim,
		MaxLen:   maxLen,
		Encoding: encoding,
	}
}

// NewTransformerWithTensors creates a new transformer model with tensor-based parameters
func NewTransformerWithTensors(vocabSize, embeddingDim, numLayers, numHeads, ffnHiddenDim, maxLen int) *TransformerWithTensors {
	// Create encoder layers
	encoder := make([]*EncoderLayerWithTensors, numLayers)
	for i := 0; i < numLayers; i++ {
		encoder[i] = NewEncoderLayerWithTensors(embeddingDim, ffnHiddenDim, numHeads)
	}
	
	// Create decoder layers
	decoder := make([]*DecoderLayerWithTensors, numLayers)
	for i := 0; i < numLayers; i++ {
		decoder[i] = NewDecoderLayerWithTensors(embeddingDim, ffnHiddenDim, numHeads)
	}
	
	return &TransformerWithTensors{
		Encoder:           encoder,
		Decoder:           decoder,
		PositionalEncoder: NewPositionalEncoding(embeddingDim, maxLen),
		EmbeddingDim:      embeddingDim,
		VocabSize:         vocabSize,
		EmbeddingMatrix:   NewRandomTensor(vocabSize, embeddingDim, true),
		OutputMatrix:      NewRandomTensor(embeddingDim, vocabSize, true),
	}
}

// NewEncoderLayerWithTensors creates a new encoder layer with tensor-based parameters
func NewEncoderLayerWithTensors(modelDim, ffnHiddenDim, numHeads int) *EncoderLayerWithTensors {
	return &EncoderLayerWithTensors{
		SelfAttention:    NewMultiHeadAttentionWithTensors(numHeads, modelDim),
		FeedForward:      NewFeedForwardWithTensors(modelDim, ffnHiddenDim),
		Norm1:            NewLayerNormWithTensors(modelDim),
		Norm2:            NewLayerNormWithTensors(modelDim),
		DropoutAttention: NewDropout(0.1),
		DropoutFFN:       NewDropout(0.1),
		DropoutResidual:  NewDropout(0.1),
	}
}

// NewDecoderLayerWithTensors creates a new decoder layer with tensor-based parameters
func NewDecoderLayerWithTensors(modelDim, ffnHiddenDim, numHeads int) *DecoderLayerWithTensors {
	return &DecoderLayerWithTensors{
		SelfAttention:     NewMultiHeadAttentionWithTensors(numHeads, modelDim),
		CrossAttention:    NewMultiHeadAttentionWithTensors(numHeads, modelDim),
		FeedForward:       NewFeedForwardWithTensors(modelDim, ffnHiddenDim),
		Norm1:             NewLayerNormWithTensors(modelDim),
		Norm2:             NewLayerNormWithTensors(modelDim),
		Norm3:             NewLayerNormWithTensors(modelDim),
		DropoutSelfAttn:   NewDropout(0.1),
		DropoutCrossAttn:  NewDropout(0.1),
		DropoutFFN:        NewDropout(0.1),
		DropoutResidual1:  NewDropout(0.1),
		DropoutResidual2:  NewDropout(0.1),
	}
}

// NewMultiHeadAttentionWithTensors creates a new multi-head attention layer with tensor-based parameters
func NewMultiHeadAttentionWithTensors(numHeads, modelDim int) *MultiHeadAttentionWithTensors {
	headDim := modelDim / numHeads
	
	return &MultiHeadAttentionWithTensors{
		NumHeads:    numHeads,
		ModelDim:    modelDim,
		HeadDim:     headDim,
		QueryWeight: NewRandomTensor(modelDim, modelDim, true),
		KeyWeight:   NewRandomTensor(modelDim, modelDim, true),
		ValueWeight: NewRandomTensor(modelDim, modelDim, true),
		OutputWeight: NewRandomTensor(modelDim, modelDim, true),
	}
}

// NewFeedForwardWithTensors creates a new feed-forward network with tensor-based parameters
func NewFeedForwardWithTensors(inputDim, hiddenDim int) *FeedForwardWithTensors {
	return &FeedForwardWithTensors{
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
		W1:        NewRandomTensor(inputDim, hiddenDim, true),
		B1:        NewZerosTensor(1, hiddenDim, true),
		W2:        NewRandomTensor(hiddenDim, inputDim, true),
		B2:        NewZerosTensor(1, inputDim, true),
	}
}

// NewLayerNormWithTensors creates a new layer normalization with tensor-based parameters
func NewLayerNormWithTensors(dim int) *LayerNormWithTensors {
	gamma := NewZerosTensor(1, dim, true)
	beta := NewZerosTensor(1, dim, true)
	
	// Initialize gamma to ones
	for j := 0; j < dim; j++ {
		gamma.Data.Data[0][j] = 1.0
	}
	
	return &LayerNormWithTensors{
		Dim:   dim,
		Gamma: gamma,
		Beta:  beta,
		Eps:   1e-5,
	}
}

// Forward performs the multi-head attention operation with tensor-based parameters
func (mha *MultiHeadAttentionWithTensors) Forward(query, key, value *Tensor) *Tensor {
	// Project inputs to query, key, value
	q := TensorMatMul(query, mha.QueryWeight)
	k := TensorMatMul(key, mha.KeyWeight)
	v := TensorMatMul(value, mha.ValueWeight)
	
	// Simple scaled dot-product attention
	// For simplicity, we're not splitting into multiple heads in this implementation
	kT := Transpose(k.Data)
	kTensor := NewTensor(kT, k.Requires)
	scores := TensorMatMul(q, kTensor)
	
	// Scale
	scaleFactor := math.Sqrt(float64(mha.ModelDim))
	for i := 0; i < scores.Data.Rows; i++ {
		for j := 0; j < scores.Data.Cols; j++ {
			scores.Data.Data[i][j] /= scaleFactor
		}
	}
	
	// Apply softmax
	attentionWeights := TensorSoftmax(scores)
	
	// Apply attention weights
	output := TensorMatMul(attentionWeights, v)
	
	// Project back to model dimension
	return TensorMatMul(output, mha.OutputWeight)
}

// Forward performs the feed-forward operation with tensor-based parameters
func (ff *FeedForwardWithTensors) Forward(input *Tensor) *Tensor {
	// First linear transformation
	hidden := TensorMatMul(input, ff.W1)
	
	// Add bias
	for i := 0; i < hidden.Data.Rows; i++ {
		for j := 0; j < hidden.Data.Cols; j++ {
			hidden.Data.Data[i][j] += ff.B1.Data.Data[0][j]
		}
	}
	
	// Apply ReLU activation
	hidden = TensorReLU(hidden)
	
	// Second linear transformation
	output := TensorMatMul(hidden, ff.W2)
	
	// Add bias
	for i := 0; i < output.Data.Rows; i++ {
		for j := 0; j < output.Data.Cols; j++ {
			output.Data.Data[i][j] += ff.B2.Data.Data[0][j]
		}
	}
	
	return output
}

// Forward performs layer normalization with tensor-based parameters
func (ln *LayerNormWithTensors) Forward(input *Tensor) *Tensor {
	result := NewZerosTensor(input.Data.Rows, input.Data.Cols, input.Requires)
	
	// Normalize each row independently
	for i := 0; i < input.Data.Rows; i++ {
		// Calculate mean
		mean := 0.0
		for j := 0; j < input.Data.Cols; j++ {
			mean += input.Data.Data[i][j]
		}
		mean /= float64(input.Data.Cols)
		
		// Calculate variance
		variance := 0.0
		for j := 0; j < input.Data.Cols; j++ {
			diff := input.Data.Data[i][j] - mean
			variance += diff * diff
		}
		variance /= float64(input.Data.Cols)
		
		// Normalize, scale, and shift
		for j := 0; j < input.Data.Cols; j++ {
			normalized := (input.Data.Data[i][j] - mean) / math.Sqrt(variance+ln.Eps)
			result.Data.Data[i][j] = normalized*ln.Gamma.Data.Data[0][j] + ln.Beta.Data.Data[0][j]
		}
	}
	
	return result
}

// Forward processes input through the encoder layer with tensor-based parameters
func (el *EncoderLayerWithTensors) Forward(input *Tensor, isTraining bool) *Tensor {
	// Self-attention
	normalized1 := el.Norm1.Forward(input)
	attnOut := el.SelfAttention.Forward(normalized1, normalized1, normalized1)
	
	// Apply dropout to attention output if training
	if isTraining {
		for i := 0; i < attnOut.Data.Rows; i++ {
			for j := 0; j < attnOut.Data.Cols; j++ {
				if el.DropoutAttention.ShouldDrop() {
					attnOut.Data.Data[i][j] = 0
				} else {
					attnOut.Data.Data[i][j] /= (1.0 - el.DropoutAttention.Rate)
				}
			}
		}
	}
	
	// Add residual connection
	residual1 := TensorAdd(input, attnOut)
	
	// Apply dropout to residual if training
	if isTraining {
		for i := 0; i < residual1.Data.Rows; i++ {
			for j := 0; j < residual1.Data.Cols; j++ {
				if el.DropoutResidual.ShouldDrop() {
					residual1.Data.Data[i][j] = 0
				} else {
					residual1.Data.Data[i][j] /= (1.0 - el.DropoutResidual.Rate)
				}
			}
		}
	}
	
	// Feed-forward network
	normalized2 := el.Norm2.Forward(residual1)
	ffnOut := el.FeedForward.Forward(normalized2)
	
	// Apply dropout to FFN output if training
	if isTraining {
		for i := 0; i < ffnOut.Data.Rows; i++ {
			for j := 0; j < ffnOut.Data.Cols; j++ {
				if el.DropoutFFN.ShouldDrop() {
					ffnOut.Data.Data[i][j] = 0
				} else {
					ffnOut.Data.Data[i][j] /= (1.0 - el.DropoutFFN.Rate)
				}
			}
		}
	}
	
	// Add residual connection
	return TensorAdd(residual1, ffnOut)
}

// Forward processes input through the decoder layer with tensor-based parameters
func (dl *DecoderLayerWithTensors) Forward(input *Tensor, encoderOutput *Tensor, isTraining bool) *Tensor {
	// Self-attention
	normalized1 := dl.Norm1.Forward(input)
	selfAttnOut := dl.SelfAttention.Forward(normalized1, normalized1, normalized1)
	
	// Apply dropout to self-attention output if training
	if isTraining {
		for i := 0; i < selfAttnOut.Data.Rows; i++ {
			for j := 0; j < selfAttnOut.Data.Cols; j++ {
				if dl.DropoutSelfAttn.ShouldDrop() {
					selfAttnOut.Data.Data[i][j] = 0
				} else {
					selfAttnOut.Data.Data[i][j] /= (1.0 - dl.DropoutSelfAttn.Rate)
				}
			}
		}
	}
	
	// Add residual connection
	residual1 := TensorAdd(input, selfAttnOut)
	
	// Apply dropout to residual if training
	if isTraining {
		for i := 0; i < residual1.Data.Rows; i++ {
			for j := 0; j < residual1.Data.Cols; j++ {
				if dl.DropoutResidual1.ShouldDrop() {
					residual1.Data.Data[i][j] = 0
				} else {
					residual1.Data.Data[i][j] /= (1.0 - dl.DropoutResidual1.Rate)
				}
			}
		}
	}
	
	// Cross-attention
	normalized2 := dl.Norm2.Forward(residual1)
	crossAttnOut := dl.CrossAttention.Forward(normalized2, encoderOutput, encoderOutput)
	
	// Apply dropout to cross-attention output if training
	if isTraining {
		for i := 0; i < crossAttnOut.Data.Rows; i++ {
			for j := 0; j < crossAttnOut.Data.Cols; j++ {
				if dl.DropoutCrossAttn.ShouldDrop() {
					crossAttnOut.Data.Data[i][j] = 0
				} else {
					crossAttnOut.Data.Data[i][j] /= (1.0 - dl.DropoutCrossAttn.Rate)
				}
			}
		}
	}
	
	// Add residual connection
	residual2 := TensorAdd(residual1, crossAttnOut)
	
	// Apply dropout to residual if training
	if isTraining {
		for i := 0; i < residual2.Data.Rows; i++ {
			for j := 0; j < residual2.Data.Cols; j++ {
				if dl.DropoutResidual2.ShouldDrop() {
					residual2.Data.Data[i][j] = 0
				} else {
					residual2.Data.Data[i][j] /= (1.0 - dl.DropoutResidual2.Rate)
				}
			}
		}
	}
	
	// Feed-forward network
	normalized3 := dl.Norm3.Forward(residual2)
	ffnOut := dl.FeedForward.Forward(normalized3)
	
	// Apply dropout to FFN output if training
	if isTraining {
		for i := 0; i < ffnOut.Data.Rows; i++ {
			for j := 0; j < ffnOut.Data.Cols; j++ {
				if dl.DropoutFFN.ShouldDrop() {
					ffnOut.Data.Data[i][j] = 0
				} else {
					ffnOut.Data.Data[i][j] /= (1.0 - dl.DropoutFFN.Rate)
				}
			}
		}
	}
	
	// Add residual connection
	return TensorAdd(residual2, ffnOut)
}

// Embed converts token indices to embeddings with tensor-based parameters
func (t *TransformerWithTensors) Embed(indices []int) *Tensor {
	result := NewZerosTensor(len(indices), t.EmbeddingDim, true)
	
	for i, idx := range indices {
		if idx >= 0 && idx < t.VocabSize {
			for j := 0; j < t.EmbeddingDim; j++ {
				result.Data.Data[i][j] = t.EmbeddingMatrix.Data.Data[idx][j]
			}
		}
	}
	
	return result
}

// Forward processes input through the transformer model with tensor-based parameters
func (t *TransformerWithTensors) Forward(srcIndices, tgtIndices []int) *Tensor {
	// Convert token indices to embeddings
	srcEmbeddings := t.Embed(srcIndices)
	tgtEmbeddings := t.Embed(tgtIndices)
	
	// Add positional encoding
	srcEmbeddings = t.AddPositionalEncoding(srcEmbeddings)
	tgtEmbeddings = t.AddPositionalEncoding(tgtEmbeddings)
	
	// Process through encoder
	encoderOutput := srcEmbeddings
	for _, layer := range t.Encoder {
		encoderOutput = layer.Forward(encoderOutput, true)
	}
	
	// Process through decoder
	decoderOutput := tgtEmbeddings
	for _, layer := range t.Decoder {
		decoderOutput = layer.Forward(decoderOutput, encoderOutput, true)
	}
	
	// Project to vocabulary size
	logits := TensorMatMul(decoderOutput, t.OutputMatrix)
	
	return logits
}

// ForwardEval processes input through the transformer model without gradient tracking
func (t *TransformerWithTensors) ForwardEval(srcIndices, tgtIndices []int) *Matrix {
	// Convert token indices to embeddings
	srcEmbeddings := t.Embed(srcIndices)
	tgtEmbeddings := t.Embed(tgtIndices)
	
	// Add positional encoding
	srcEmbeddings = t.AddPositionalEncoding(srcEmbeddings)
	tgtEmbeddings = t.AddPositionalEncoding(tgtEmbeddings)
	
	// Process through encoder
	encoderOutput := srcEmbeddings
	for _, layer := range t.Encoder {
		encoderOutput = layer.Forward(encoderOutput, false)
	}
	
	// Process through decoder
	decoderOutput := tgtEmbeddings
	for _, layer := range t.Decoder {
		decoderOutput = layer.Forward(decoderOutput, encoderOutput, false)
	}
	
	// Project to vocabulary size
	logits := TensorMatMul(decoderOutput, t.OutputMatrix)
	
	// Apply softmax to get probabilities
	probs := TensorSoftmax(logits)
	
	return probs.Data
}

// AddPositionalEncoding adds positional encoding to the input embeddings
func (t *TransformerWithTensors) AddPositionalEncoding(embeddings *Tensor) *Tensor {
	if embeddings.Data.Rows > t.PositionalEncoder.MaxLen {
		panic("Input sequence length exceeds maximum length for positional encoding")
	}
	
	result := NewZerosTensor(embeddings.Data.Rows, embeddings.Data.Cols, embeddings.Requires)
	
	for i := 0; i < embeddings.Data.Rows; i++ {
		for j := 0; j < embeddings.Data.Cols; j++ {
			result.Data.Data[i][j] = embeddings.Data.Data[i][j] + t.PositionalEncoder.Encoding.Data[i][j]
		}
	}
	
	return result
}

// GetParameters returns all trainable parameters of the transformer model
func (t *TransformerWithTensors) GetParameters() map[string]*Tensor {
	params := make(map[string]*Tensor)
	
	// Embedding and output matrices
	params["embedding"] = t.EmbeddingMatrix
	params["output"] = t.OutputMatrix
	
	// Encoder parameters
	for i, layer := range t.Encoder {
		// Self-attention
		params[fmt.Sprintf("encoder_%d_self_query", i)] = layer.SelfAttention.QueryWeight
		params[fmt.Sprintf("encoder_%d_self_key", i)] = layer.SelfAttention.KeyWeight
		params[fmt.Sprintf("encoder_%d_self_value", i)] = layer.SelfAttention.ValueWeight
		params[fmt.Sprintf("encoder_%d_self_output", i)] = layer.SelfAttention.OutputWeight
		
		// Layer normalization
		params[fmt.Sprintf("encoder_%d_norm1_gamma", i)] = layer.Norm1.Gamma
		params[fmt.Sprintf("encoder_%d_norm1_beta", i)] = layer.Norm1.Beta
		params[fmt.Sprintf("encoder_%d_norm2_gamma", i)] = layer.Norm2.Gamma
		params[fmt.Sprintf("encoder_%d_norm2_beta", i)] = layer.Norm2.Beta
		
		// Feed-forward
		params[fmt.Sprintf("encoder_%d_ffn_w1", i)] = layer.FeedForward.W1
		params[fmt.Sprintf("encoder_%d_ffn_b1", i)] = layer.FeedForward.B1
		params[fmt.Sprintf("encoder_%d_ffn_w2", i)] = layer.FeedForward.W2
		params[fmt.Sprintf("encoder_%d_ffn_b2", i)] = layer.FeedForward.B2
	}
	
	// Decoder parameters
	for i, layer := range t.Decoder {
		// Self-attention
		params[fmt.Sprintf("decoder_%d_self_query", i)] = layer.SelfAttention.QueryWeight
		params[fmt.Sprintf("decoder_%d_self_key", i)] = layer.SelfAttention.KeyWeight
		params[fmt.Sprintf("decoder_%d_self_value", i)] = layer.SelfAttention.ValueWeight
		params[fmt.Sprintf("decoder_%d_self_output", i)] = layer.SelfAttention.OutputWeight
		
		// Cross-attention
		params[fmt.Sprintf("decoder_%d_cross_query", i)] = layer.CrossAttention.QueryWeight
		params[fmt.Sprintf("decoder_%d_cross_key", i)] = layer.CrossAttention.KeyWeight
		params[fmt.Sprintf("decoder_%d_cross_value", i)] = layer.CrossAttention.ValueWeight
		params[fmt.Sprintf("decoder_%d_cross_output", i)] = layer.CrossAttention.OutputWeight
		
		// Layer normalization
		params[fmt.Sprintf("decoder_%d_norm1_gamma", i)] = layer.Norm1.Gamma
		params[fmt.Sprintf("decoder_%d_norm1_beta", i)] = layer.Norm1.Beta
		params[fmt.Sprintf("decoder_%d_norm2_gamma", i)] = layer.Norm2.Gamma
		params[fmt.Sprintf("decoder_%d_norm2_beta", i)] = layer.Norm2.Beta
		params[fmt.Sprintf("decoder_%d_norm3_gamma", i)] = layer.Norm3.Gamma
		params[fmt.Sprintf("decoder_%d_norm3_beta", i)] = layer.Norm3.Beta
		
		// Feed-forward
		params[fmt.Sprintf("decoder_%d_ffn_w1", i)] = layer.FeedForward.W1
		params[fmt.Sprintf("decoder_%d_ffn_b1", i)] = layer.FeedForward.B1
		params[fmt.Sprintf("decoder_%d_ffn_w2", i)] = layer.FeedForward.W2
		params[fmt.Sprintf("decoder_%d_ffn_b2", i)] = layer.FeedForward.B2
	}
	
	return params
}
