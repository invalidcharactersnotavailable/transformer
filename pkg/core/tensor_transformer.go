package transformer

import (
	"fmt"
	"math"
	// "github.com/transformer_reorganized/pkg/moe" // Keep for now, might be used elsewhere or remove later if truly unused
	// fmt is not used if LoadTransformerWithTensors is commented out.
)

// MoELayer represents a Mixture of Experts layer (simplified for core package)
// This struct will eventually have its own parameters using pkg/core.Tensor
type MoELayer struct {
	// Example: Experts []*Expert // Where Expert is another local struct using pkg/core.Tensor
	// For now, it's empty as per plan.
}

func (moeL *MoELayer) GetParameters() []*Tensor {
	// Placeholder: In a real MoE, would collect parameters from experts and router
	return []*Tensor{}
}

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
	IsMoE            bool
	MoELayer         *MoELayer // Changed to local type
}

func (el *EncoderLayerWithTensors) GetParameters() []*Tensor {
	var params []*Tensor
	if el.SelfAttention != nil {
		params = append(params, el.SelfAttention.GetParameters()...)
	}
	if el.Norm1 != nil {
		params = append(params, el.Norm1.GetParameters()...)
	}
	if el.Norm2 != nil {
		params = append(params, el.Norm2.GetParameters()...)
	}
	if el.IsMoE && el.MoELayer != nil {
		params = append(params, el.MoELayer.GetParameters()...)
	} else if el.FeedForward != nil {
		params = append(params, el.FeedForward.GetParameters()...)
	}
	return params
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
	IsMoE             bool
	MoELayer          *MoELayer // Changed to local type
}

func (dl *DecoderLayerWithTensors) GetParameters() []*Tensor {
	var params []*Tensor
	if dl.SelfAttention != nil {
		params = append(params, dl.SelfAttention.GetParameters()...)
	}
	if dl.CrossAttention != nil {
		params = append(params, dl.CrossAttention.GetParameters()...)
	}
	if dl.Norm1 != nil {
		params = append(params, dl.Norm1.GetParameters()...)
	}
	if dl.Norm2 != nil {
		params = append(params, dl.Norm2.GetParameters()...)
	}
	if dl.Norm3 != nil {
		params = append(params, dl.Norm3.GetParameters()...)
	}
	if dl.IsMoE && dl.MoELayer != nil {
		params = append(params, dl.MoELayer.GetParameters()...)
	} else if dl.FeedForward != nil {
		params = append(params, dl.FeedForward.GetParameters()...)
	}
	return params
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

func (mha *MultiHeadAttentionWithTensors) GetParameters() []*Tensor {
	params := []*Tensor{}
	if mha.QueryWeight != nil {
		params = append(params, mha.QueryWeight)
	}
	if mha.KeyWeight != nil {
		params = append(params, mha.KeyWeight)
	}
	if mha.ValueWeight != nil {
		params = append(params, mha.ValueWeight)
	}
	if mha.OutputWeight != nil {
		params = append(params, mha.OutputWeight)
	}
	return params
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

func (ffn *FeedForwardWithTensors) GetParameters() []*Tensor {
	params := []*Tensor{}
	if ffn.W1 != nil {
		params = append(params, ffn.W1)
	}
	if ffn.B1 != nil {
		params = append(params, ffn.B1)
	}
	if ffn.W2 != nil {
		params = append(params, ffn.W2)
	}
	if ffn.B2 != nil {
		params = append(params, ffn.B2)
	}
	return params
}

// LayerNormWithTensors represents layer normalization with tensor-based parameters
type LayerNormWithTensors struct {
	Dim    int
	Gamma  *Tensor
	Beta   *Tensor
	Eps    float64
}

func (ln *LayerNormWithTensors) GetParameters() []*Tensor {
	params := []*Tensor{}
	if ln.Gamma != nil {
		params = append(params, ln.Gamma)
	}
	if ln.Beta != nil {
		params = append(params, ln.Beta)
	}
	return params
}

// NewTransformerWithTensors creates a new transformer model with tensor-based parameters
func NewTransformerWithTensors(vocabSize, embeddingDim, numLayers, numHeads, ffnHiddenDim, maxLen int) *TransformerWithTensors {
	// Create encoder layers
	encoder := make([]*EncoderLayerWithTensors, numLayers)
	for i := 0; i < numLayers; i++ {
		// Defaulting useMoE to false for now
		encoder[i] = NewEncoderLayerWithTensors(embeddingDim, ffnHiddenDim, numHeads, false)
	}
	
	// Create decoder layers
	decoder := make([]*DecoderLayerWithTensors, numLayers)
	for i := 0; i < numLayers; i++ {
		// Defaulting useMoE to false for now
		decoder[i] = NewDecoderLayerWithTensors(embeddingDim, ffnHiddenDim, numHeads, false)
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
func NewEncoderLayerWithTensors(modelDim, ffnHiddenDim, numHeads int, useMoE bool) *EncoderLayerWithTensors {
	el := &EncoderLayerWithTensors{
		SelfAttention:    NewMultiHeadAttentionWithTensors(numHeads, modelDim),
		Norm1:            NewLayerNormWithTensors(modelDim),
		Norm2:            NewLayerNormWithTensors(modelDim),
		DropoutAttention: NewDropout(0.1),
		DropoutFFN:       NewDropout(0.1),
		DropoutResidual:  NewDropout(0.1),
		IsMoE:            useMoE,
	}
	if useMoE {
		el.MoELayer = &MoELayer{} // Changed to local type
		el.FeedForward = nil
	} else {
		el.FeedForward = NewFeedForwardWithTensors(modelDim, ffnHiddenDim)
		el.MoELayer = nil
	}
	return el
}

// NewDecoderLayerWithTensors creates a new decoder layer with tensor-based parameters
func NewDecoderLayerWithTensors(modelDim, ffnHiddenDim, numHeads int, useMoE bool) *DecoderLayerWithTensors {
	dl := &DecoderLayerWithTensors{
		SelfAttention:     NewMultiHeadAttentionWithTensors(numHeads, modelDim),
		CrossAttention:    NewMultiHeadAttentionWithTensors(numHeads, modelDim),
		Norm1:             NewLayerNormWithTensors(modelDim),
		Norm2:             NewLayerNormWithTensors(modelDim),
		Norm3:             NewLayerNormWithTensors(modelDim),
		DropoutSelfAttn:   NewDropout(0.1),
		DropoutCrossAttn:  NewDropout(0.1),
		DropoutFFN:        NewDropout(0.1),
		DropoutResidual1:  NewDropout(0.1),
		DropoutResidual2:  NewDropout(0.1),
		IsMoE:             useMoE,
	}
	if useMoE {
		dl.MoELayer = &MoELayer{} // Changed to local type
		dl.FeedForward = nil
	} else {
		dl.FeedForward = NewFeedForwardWithTensors(modelDim, ffnHiddenDim)
		dl.MoELayer = nil
	}
	return dl
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

// GetParameters returns all trainable parameters of the transformer model as a slice.
func (t *TransformerWithTensors) GetParameters() []*Tensor {
	paramsMap := make(map[*Tensor]bool) // To avoid duplicates
	var params []*Tensor

	add := func(tensor *Tensor) {
		if tensor != nil && !paramsMap[tensor] { // Assuming all tensors returned by component GetParameters are relevant
			params = append(params, tensor)
			paramsMap[tensor] = true
		}
	}

	addMultiple := func(tensors []*Tensor) {
		for _, tensor := range tensors {
			add(tensor)
		}
	}

	// Embedding and output matrices
	add(t.EmbeddingMatrix)
	add(t.OutputMatrix)

	// Encoder parameters
	for _, layer := range t.Encoder {
		if layer != nil { // Add nil check for layer
			addMultiple(layer.GetParameters()) // Assumes layer.GetParameters() exists
		}
	}

	// Decoder parameters
	for _, layer := range t.Decoder {
		if layer != nil { // Add nil check for layer
			addMultiple(layer.GetParameters()) // Assumes layer.GetParameters() exists
		}
	}

	return params
}

// GetMoELayers returns all MoE layers in the transformer
func (t *TransformerWithTensors) GetMoELayers() []*MoELayer { // Changed to local type
	layers := []*MoELayer{} // Changed to local type
	for _, encoderLayer := range t.Encoder {
		if encoderLayer.IsMoE && encoderLayer.MoELayer != nil {
			layers = append(layers, encoderLayer.MoELayer)
		}
	}
	for _, decoderLayer := range t.Decoder {
		if decoderLayer.IsMoE && decoderLayer.MoELayer != nil {
			layers = append(layers, decoderLayer.MoELayer) // This should now correctly append the local MoELayer type
		}
	}
	return layers
}

// ModelSerializer handles saving and loading transformer models
// type ModelSerializer struct{}

// NewModelSerializer creates a new model serializer
// func NewModelSerializer() *ModelSerializer {
// 	return &ModelSerializer{}
// }

// SaveTransformerWithTensors saves a transformer model to disk
// func (ms *ModelSerializer) SaveTransformerWithTensors(model *TransformerWithTensors, path string) error {
// 	// In a real implementation, this would serialize the model to disk
// 	// For this example, we'll just simulate successful saving
// 	return nil
// }

// LoadTransformerWithTensors loads a transformer model from disk
// func (ms *ModelSerializer) LoadTransformerWithTensors(path string) (*TransformerWithTensors, error) {
// 	// In a real implementation, this would deserialize the model from disk
// 	// For this example, we'll just return nil and an error
// 	return nil, fmt.Errorf("not implemented")
// }

// TrainingExample represents a single training example
// type TrainingExample struct {
// 	SourceTokens []int
// 	TargetTokens []int
// 	Labels       []int
// }

// NewTrainingExample creates a new training example
// func NewTrainingExample(sourceTokens, targetTokens, labels []int) *TrainingExample {
// 	return &TrainingExample{
// 		SourceTokens: sourceTokens,
// 		TargetTokens: targetTokens,
// 		Labels:       labels,
// 	}
// }
