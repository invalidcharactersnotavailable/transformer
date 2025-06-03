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
	AttentionDropoutRate float64
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
	"coreconfig "github.com/transformer_reorganized/pkg/core" // Alias for core config
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
	Config            *coreconfig.Config // Store config for GetParameters
}

// NewTransformerWithTensors creates a new transformer model with tensor-based parameters
func NewTransformerWithTensors(config *coreconfig.Config) *TransformerWithTensors {
	encoder := make([]*EncoderLayerWithTensors, config.NumLayers)
	if config.UseCrossLayerParameterSharing && config.NumLayers > 0 {
		sharedEncoderLayer := NewEncoderLayerWithTensors(config.EmbeddingDim, config.FFNHiddenDim, config.NumHeads)
		for i := 0; i < config.NumLayers; i++ {
			encoder[i] = sharedEncoderLayer
		}
	} else {
		for i := 0; i < config.NumLayers; i++ {
			encoder[i] = NewEncoderLayerWithTensors(config.EmbeddingDim, config.FFNHiddenDim, config.NumHeads)
		}
	}
	
	// Assuming similar structure for decoder layers if they exist (e.g. config.NumDecoderLayers)
	// For this example, let's assume a symmetric number of decoder layers as encoder layers for simplicity.
	decoder := make([]*DecoderLayerWithTensors, config.NumLayers) // Or config.NumDecoderLayers
	if config.UseCrossLayerParameterSharing && config.NumLayers > 0 { // Or config.NumDecoderLayers
		// Potentially a *different* shared decoder layer than the encoder one
		sharedDecoderLayer := NewDecoderLayerWithTensors(config.EmbeddingDim, config.FFNHiddenDim, config.NumHeads)
		for i := 0; i < config.NumLayers; i++ { // Or config.NumDecoderLayers
			decoder[i] = sharedDecoderLayer
		}
	} else {
		for i := 0; i < config.NumLayers; i++ { // Or config.NumDecoderLayers
			decoder[i] = NewDecoderLayerWithTensors(config.EmbeddingDim, config.FFNHiddenDim, config.NumHeads)
		}
	}
	
	embeddingMatrix, _ := NewRandomTensor(config.VocabSize, config.EmbeddingDim, &TensorConfig{RequiresGrad: true, Name: "embedding_matrix"})
	outputMatrix, _ := NewRandomTensor(config.EmbeddingDim, config.VocabSize, &TensorConfig{RequiresGrad: true, Name: "output_matrix"})


	return &TransformerWithTensors{
		Encoder:           encoder,
		Decoder:           decoder,
		PositionalEncoder: NewPositionalEncoding(config.EmbeddingDim, config.MaxLen),
		EmbeddingDim:      config.EmbeddingDim,
		VocabSize:         config.VocabSize,
		EmbeddingMatrix:   embeddingMatrix,
		OutputMatrix:      outputMatrix,
		Config:            config, // Store the config
	}
}

// NewEncoderLayerWithTensors creates a new encoder layer with tensor-based parameters
func NewEncoderLayerWithTensors(modelDim, ffnHiddenDim, numHeads int) *EncoderLayerWithTensors {
	// Defaulting attention dropout rate to 0.1, this could be a parameter in a config struct later
	return &EncoderLayerWithTensors{
		SelfAttention:    NewMultiHeadAttentionWithTensors(numHeads, modelDim, 0.1),
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
	// Defaulting attention dropout rate to 0.1, this could be a parameter in a config struct later
	return &DecoderLayerWithTensors{
		SelfAttention:     NewMultiHeadAttentionWithTensors(numHeads, modelDim, 0.1),
		CrossAttention:    NewMultiHeadAttentionWithTensors(numHeads, modelDim, 0.1),
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
func NewMultiHeadAttentionWithTensors(numHeads, modelDim int, attentionDropoutRate float64) *MultiHeadAttentionWithTensors {
	headDim := modelDim / numHeads
	
	// Error handling for NewRandomTensor can be added here if it returns errors
	// For simplicity, assuming it panics or is error-free as per current structure
	return &MultiHeadAttentionWithTensors{
		NumHeads:    numHeads,
		ModelDim:    modelDim,
		HeadDim:     headDim,
		QueryWeight: NewRandomTensor(modelDim, modelDim, &TensorConfig{RequiresGrad: true, Name: "mha_query_w"}),
		KeyWeight:   NewRandomTensor(modelDim, modelDim, &TensorConfig{RequiresGrad: true, Name: "mha_key_w"}),
		ValueWeight: NewRandomTensor(modelDim, modelDim, &TensorConfig{RequiresGrad: true, Name: "mha_value_w"}),
		OutputWeight: NewRandomTensor(modelDim, modelDim, &TensorConfig{RequiresGrad: true, Name: "mha_output_w"}),
		AttentionDropoutRate: attentionDropoutRate,
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
func (mha *MultiHeadAttentionWithTensors) Forward(query, key, value *Tensor, mask *Tensor, isTraining bool) (*Tensor, error) {
	if query == nil || key == nil || value == nil {
		return nil, fmt.Errorf("query, key, and value tensors cannot be nil")
	}
	seqLen := query.Data.Rows // Assuming query, key, value are (seq_len, model_dim)

	// 1. Initial linear projections
	// Note: Using MatMul from pkg/autodiff/autodiff.go which handles tensors
	qFull, err := MatMul(query, mha.QueryWeight)
	if err != nil { return nil, fmt.Errorf("query projection failed: %v", err) }
	kFull, err := MatMul(key, mha.KeyWeight)
	if err != nil { return nil, fmt.Errorf("key projection failed: %v", err) }
	vFull, err := MatMul(value, mha.ValueWeight)
	if err != nil { return nil, fmt.Errorf("value projection failed: %v", err) }

	headOutputs := make([]*Tensor, 0, mha.NumHeads)
	var opErr error // To capture errors within the loop

	// 2. Process each head
	for h := 0; h < mha.NumHeads; h++ {
		startCol := h * mha.HeadDim
		headNamePrefix := fmt.Sprintf("head_%d_", h)

		qHead, err := SliceColsTensor(qFull, startCol, mha.HeadDim, headNamePrefix+"q")
		if err != nil { opErr = err; break }
		kHead, err := SliceColsTensor(kFull, startCol, mha.HeadDim, headNamePrefix+"k")
		if err != nil { opErr = err; break }
		vHead, err := SliceColsTensor(vFull, startCol, mha.HeadDim, headNamePrefix+"v")
		if err != nil { opErr = err; break }

		// Scaled Dot-Product Attention
		kHeadT, err := TensorTranspose(kHead) // Assuming TensorTranspose is from pkg/autodiff/autodiff.go
		if err != nil { opErr = err; break }

		scores, err := MatMul(qHead, kHeadT)
		if err != nil { opErr = err; break }

		scaleFactorVal := math.Sqrt(float64(mha.HeadDim))
		// Assuming ScalarMultiply is from pkg/autodiff/autodiff.go
		scaledScores, err := ScalarMultiply(scores, 1.0/scaleFactorVal)
		if err != nil { opErr = err; break }

		maskedScores := scaledScores
		if mask != nil {
			if mask.Data.Rows != seqLen || mask.Data.Cols != seqLen {
				 opErr = fmt.Errorf("attention mask dimensions (%dx%d) incompatible with sequence length %d for head %d", mask.Data.Rows, mask.Data.Cols, seqLen, h); break;
			}
			// ApplyAttentionMaskTensor expects 0 for mask, 1 for keep.
			maskedScores, err = ApplyAttentionMaskTensor(scaledScores, mask, -1e9, headNamePrefix+"masked_scores")
			if err != nil { opErr = err; break }
		}

		attentionWeights, err := Softmax(maskedScores) // Softmax is from pkg/autodiff/autodiff.go (row-wise)
		if err != nil { opErr = err; break }

		if isTraining && mha.AttentionDropoutRate > 0.0 {
			attentionWeights, err = DropoutTensor(attentionWeights, mha.AttentionDropoutRate, true, headNamePrefix+"attn_dropout")
			if err != nil { opErr = err; break }
		}

		weightedValue, err := MatMul(attentionWeights, vHead)
		if err != nil { opErr = err; break }

		headOutputs = append(headOutputs, weightedValue)
	}
	if opErr != nil { return nil, fmt.Errorf("error processing head: %v", opErr) }

	// 3. Concatenate head outputs
	if len(headOutputs) == 0 && mha.NumHeads > 0 {
		return nil, fmt.Errorf("no head outputs to concatenate, though NumHeads is %d", mha.NumHeads)
	}
    if mha.NumHeads == 0 { // Should be caught by constructor, but defensive
        return nil, fmt.Errorf("NumHeads is 0, cannot process attention")
    }


	var concatenatedOutput *Tensor
	if len(headOutputs) == 1 {
		concatenatedOutput = headOutputs[0]
	} else {
		concatenatedOutput, err = ConcatenateColsTensor(headOutputs, "concat_heads")
		if err != nil { return nil, fmt.Errorf("concatenating head outputs failed: %v", err)}
	}
	

	// 4. Final linear projection
	finalOutput, err := MatMul(concatenatedOutput, mha.OutputWeight)
	if err != nil { return nil, fmt.Errorf("final output projection failed: %v", err) }

	return finalOutput, nil
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
func (el *EncoderLayerWithTensors) Forward(input *Tensor, isTraining bool) (*Tensor, error) {
	// Self-attention
	normalized1 := el.Norm1.Forward(input) // Assuming Norm1.Forward doesn't return error or is legacy
	
	// Pass nil for mask for now, and isTraining
	attnOut, err := el.SelfAttention.Forward(normalized1, normalized1, normalized1, nil, isTraining)
	if err != nil {
		return nil, fmt.Errorf("self attention in encoder layer failed: %v", err)
	}
	
	// Dropout on attention output is now handled inside SelfAttention.Forward if AttentionDropoutRate > 0

	// Add residual connection
	// Assuming TensorAdd doesn't return error or is legacy
	residual1, err := Add(input, attnOut)
	if err != nil {
		return nil, fmt.Errorf("first residual connection in encoder layer failed: %v", err)
	}
	
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
	// Assuming TensorAdd doesn't return error or is legacy
	finalResidual, err := Add(residual1, ffnOut)
	if err != nil {
		return nil, fmt.Errorf("second residual connection in encoder layer failed: %v", err)
	}
	return finalResidual, nil
}

// Forward processes input through the decoder layer with tensor-based parameters
func (dl *DecoderLayerWithTensors) Forward(input *Tensor, encoderOutput *Tensor, isTraining bool) (*Tensor, error) {
	// Self-attention
	normalized1 := dl.Norm1.Forward(input) // Assuming Norm1.Forward is legacy or error is handled if necessary
	
	// Pass nil for mask (causal mask would be needed here in a full setup)
	selfAttnOut, err := dl.SelfAttention.Forward(normalized1, normalized1, normalized1, nil, isTraining)
	if err != nil {
		return nil, fmt.Errorf("self attention in decoder layer failed: %v", err)
	}
	
	// Dropout for self-attention is handled within SelfAttention.Forward

	// Add residual connection
	residual1, err := Add(input, selfAttnOut) // Assuming Add is the tensor op from autodiff.go
	if err != nil {
		return nil, fmt.Errorf("first residual connection in decoder layer failed: %v", err)
	}
	
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
	normalized2 := dl.Norm2.Forward(residual1) // Assuming Norm2.Forward is legacy
	
	// Pass nil for mask (encoder padding mask might be needed here)
	crossAttnOut, err := dl.CrossAttention.Forward(normalized2, encoderOutput, encoderOutput, nil, isTraining)
	if err != nil {
		return nil, fmt.Errorf("cross attention in decoder layer failed: %v", err)
	}
	
	// Dropout for cross-attention is handled within CrossAttention.Forward

	// Add residual connection
	residual2, err := Add(residual1, crossAttnOut) // Assuming Add is the tensor op
	if err != nil {
		return nil, fmt.Errorf("second residual connection in decoder layer failed: %v", err)
	}
	
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
	finalResidual, err := Add(residual2, ffnOut) // Assuming Add is the tensor op
	if err != nil {
		return nil, fmt.Errorf("third residual connection in decoder layer failed: %v", err)
	}
	return finalResidual, nil
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
	var err error
	encoderOutput := srcEmbeddings
	for i, layer := range t.Encoder {
		encoderOutput, err = layer.Forward(encoderOutput, true)
		if err != nil {
			// In a real scenario, might panic or return error if Forward itself could return error
			// For now, assuming Forward might panic or this is a simplified error path
			fmt.Printf("Error in encoder layer %d: %v\n", i, err) // Placeholder error handling
			return nil // Or handle error more gracefully
		}
	}
	
	// Process through decoder
	decoderOutput := tgtEmbeddings
	for i, layer := range t.Decoder {
		decoderOutput, err = layer.Forward(decoderOutput, encoderOutput, true)
		if err != nil {
			fmt.Printf("Error in decoder layer %d: %v\n", i, err) // Placeholder error handling
			return nil // Or handle error more gracefully
		}
	}
	
	// Project to vocabulary size
	// Assuming TensorMatMul is legacy or error is handled if necessary
	logits, err := MatMul(decoderOutput, t.OutputMatrix)
	if err != nil {
		fmt.Printf("Error in final projection: %v\n", err) // Placeholder error handling
		return nil
	}
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
	var err error
	encoderOutput := srcEmbeddings
	for i, layer := range t.Encoder {
		encoderOutput, err = layer.Forward(encoderOutput, false)
		if err != nil {
			fmt.Printf("Error in ForwardEval encoder layer %d: %v\n", i, err) // Placeholder
			return nil // Or a zero matrix
		}
	}
	
	// Process through decoder
	decoderOutput := tgtEmbeddings
	for i, layer := range t.Decoder {
		decoderOutput, err = layer.Forward(decoderOutput, encoderOutput, false)
		if err != nil {
			fmt.Printf("Error in ForwardEval decoder layer %d: %v\n", i, err) // Placeholder
			return nil // Or a zero matrix
		}
	}
	
	// Project to vocabulary size
	logits, err := MatMul(decoderOutput, t.OutputMatrix)
	if err != nil {
		fmt.Printf("Error in ForwardEval final projection: %v\n", err) // Placeholder
		return nil
	}
	
	// Apply softmax to get probabilities
	probs, err := Softmax(logits) // Assuming Softmax is the tensor op from autodiff.go
	if err != nil {
		fmt.Printf("Error in ForwardEval softmax: %v\n", err) // Placeholder
		return nil
	}
	
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
	params["embedding_matrix"] = t.EmbeddingMatrix
	params["output_matrix"] = t.OutputMatrix
	
	// Encoder parameters
	if t.Config.UseCrossLayerParameterSharing && len(t.Encoder) > 0 {
		sharedEncoderLayer := t.Encoder[0]
		params["shared_encoder_self_query"] = sharedEncoderLayer.SelfAttention.QueryWeight
		params["shared_encoder_self_key"] = sharedEncoderLayer.SelfAttention.KeyWeight
		params["shared_encoder_self_value"] = sharedEncoderLayer.SelfAttention.ValueWeight
		params["shared_encoder_self_output"] = sharedEncoderLayer.SelfAttention.OutputWeight
		params["shared_encoder_norm1_gamma"] = sharedEncoderLayer.Norm1.Gamma
		params["shared_encoder_norm1_beta"] = sharedEncoderLayer.Norm1.Beta
		params["shared_encoder_norm2_gamma"] = sharedEncoderLayer.Norm2.Gamma
		params["shared_encoder_norm2_beta"] = sharedEncoderLayer.Norm2.Beta
		params["shared_encoder_ffn_w1"] = sharedEncoderLayer.FeedForward.W1
		params["shared_encoder_ffn_b1"] = sharedEncoderLayer.FeedForward.B1
		params["shared_encoder_ffn_w2"] = sharedEncoderLayer.FeedForward.W2
		params["shared_encoder_ffn_b2"] = sharedEncoderLayer.FeedForward.B2
	} else {
		for i, layer := range t.Encoder {
			params[fmt.Sprintf("encoder_%d_self_query", i)] = layer.SelfAttention.QueryWeight
			params[fmt.Sprintf("encoder_%d_self_key", i)] = layer.SelfAttention.KeyWeight
			params[fmt.Sprintf("encoder_%d_self_value", i)] = layer.SelfAttention.ValueWeight
			params[fmt.Sprintf("encoder_%d_self_output", i)] = layer.SelfAttention.OutputWeight
			params[fmt.Sprintf("encoder_%d_norm1_gamma", i)] = layer.Norm1.Gamma
			params[fmt.Sprintf("encoder_%d_norm1_beta", i)] = layer.Norm1.Beta
			params[fmt.Sprintf("encoder_%d_norm2_gamma", i)] = layer.Norm2.Gamma
			params[fmt.Sprintf("encoder_%d_norm2_beta", i)] = layer.Norm2.Beta
			params[fmt.Sprintf("encoder_%d_ffn_w1", i)] = layer.FeedForward.W1
			params[fmt.Sprintf("encoder_%d_ffn_b1", i)] = layer.FeedForward.B1
			params[fmt.Sprintf("encoder_%d_ffn_w2", i)] = layer.FeedForward.W2
			params[fmt.Sprintf("encoder_%d_ffn_b2", i)] = layer.FeedForward.B2
		}
	}
	
	// Decoder parameters
	if t.Config.UseCrossLayerParameterSharing && len(t.Decoder) > 0 {
		sharedDecoderLayer := t.Decoder[0]
		params["shared_decoder_self_query"] = sharedDecoderLayer.SelfAttention.QueryWeight
		params["shared_decoder_self_key"] = sharedDecoderLayer.SelfAttention.KeyWeight
		params["shared_decoder_self_value"] = sharedDecoderLayer.SelfAttention.ValueWeight
		params["shared_decoder_self_output"] = sharedDecoderLayer.SelfAttention.OutputWeight
		params["shared_decoder_cross_query"] = sharedDecoderLayer.CrossAttention.QueryWeight
		params["shared_decoder_cross_key"] = sharedDecoderLayer.CrossAttention.KeyWeight
		params["shared_decoder_cross_value"] = sharedDecoderLayer.CrossAttention.ValueWeight
		params["shared_decoder_cross_output"] = sharedDecoderLayer.CrossAttention.OutputWeight
		params["shared_decoder_norm1_gamma"] = sharedDecoderLayer.Norm1.Gamma
		params["shared_decoder_norm1_beta"] = sharedDecoderLayer.Norm1.Beta
		params["shared_decoder_norm2_gamma"] = sharedDecoderLayer.Norm2.Gamma
		params["shared_decoder_norm2_beta"] = sharedDecoderLayer.Norm2.Beta
		params["shared_decoder_norm3_gamma"] = sharedDecoderLayer.Norm3.Gamma
		params["shared_decoder_norm3_beta"] = sharedDecoderLayer.Norm3.Beta
		params["shared_decoder_ffn_w1"] = sharedDecoderLayer.FeedForward.W1
		params["shared_decoder_ffn_b1"] = sharedDecoderLayer.FeedForward.B1
		params["shared_decoder_ffn_w2"] = sharedDecoderLayer.FeedForward.W2
		params["shared_decoder_ffn_b2"] = sharedDecoderLayer.FeedForward.B2
	} else {
		for i, layer := range t.Decoder {
			params[fmt.Sprintf("decoder_%d_self_query", i)] = layer.SelfAttention.QueryWeight
			params[fmt.Sprintf("decoder_%d_self_key", i)] = layer.SelfAttention.KeyWeight
			params[fmt.Sprintf("decoder_%d_self_value", i)] = layer.SelfAttention.ValueWeight
			params[fmt.Sprintf("decoder_%d_self_output", i)] = layer.SelfAttention.OutputWeight
			params[fmt.Sprintf("decoder_%d_cross_query", i)] = layer.CrossAttention.QueryWeight
			params[fmt.Sprintf("decoder_%d_cross_key", i)] = layer.CrossAttention.KeyWeight
			params[fmt.Sprintf("decoder_%d_cross_value", i)] = layer.CrossAttention.ValueWeight
			params[fmt.Sprintf("decoder_%d_cross_output", i)] = layer.CrossAttention.OutputWeight
			params[fmt.Sprintf("decoder_%d_norm1_gamma", i)] = layer.Norm1.Gamma
			params[fmt.Sprintf("decoder_%d_norm1_beta", i)] = layer.Norm1.Beta
			params[fmt.Sprintf("decoder_%d_norm2_gamma", i)] = layer.Norm2.Gamma
			params[fmt.Sprintf("decoder_%d_norm2_beta", i)] = layer.Norm2.Beta
			params[fmt.Sprintf("decoder_%d_norm3_gamma", i)] = layer.Norm3.Gamma
			params[fmt.Sprintf("decoder_%d_norm3_beta", i)] = layer.Norm3.Beta
			params[fmt.Sprintf("decoder_%d_ffn_w1", i)] = layer.FeedForward.W1
			params[fmt.Sprintf("decoder_%d_ffn_b1", i)] = layer.FeedForward.B1
			params[fmt.Sprintf("decoder_%d_ffn_w2", i)] = layer.FeedForward.W2
			params[fmt.Sprintf("decoder_%d_ffn_b2", i)] = layer.FeedForward.B2
		}
	}
	
	return params
}
