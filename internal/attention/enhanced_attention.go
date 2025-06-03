package transformer

// AdvancedAttention represents an enhanced multi-head attention mechanism
type AdvancedAttention struct {
	NumHeads     int
	ModelDim     int
	HeadDim      int
	QueryWeight  *Matrix
	KeyWeight    *Matrix
	ValueWeight  *Matrix
	OutputWeight *Matrix
	AttentionDropout *Dropout
}

// NewAdvancedAttention creates a new advanced attention layer
func NewAdvancedAttention(numHeads, modelDim int, dropoutRate float64) *AdvancedAttention {
	headDim := modelDim / numHeads
	
	return &AdvancedAttention{
		NumHeads:     numHeads,
		ModelDim:     modelDim,
		HeadDim:      headDim,
		QueryWeight:  NewRandomMatrix(modelDim, modelDim),
		KeyWeight:    NewRandomMatrix(modelDim, modelDim),
		ValueWeight:  NewRandomMatrix(modelDim, modelDim),
		OutputWeight: NewRandomMatrix(modelDim, modelDim),
		AttentionDropout: NewDropout(dropoutRate),
	}
}

// Forward performs the advanced multi-head attention operation
func (aa *AdvancedAttention) Forward(query, key, value *Matrix, mask *AttentionMask, isTraining bool) *Matrix {
	// Project inputs to query, key, value
	q := MatMul(query, aa.QueryWeight)
	k := MatMul(key, aa.KeyWeight)
	v := MatMul(value, aa.ValueWeight)
	
	// Scaled dot-product attention
	kT := Transpose(k)
	scores := MatMul(q, kT)
	
	// Scale
	scaleFactor := float64(aa.HeadDim)
	for i := 0; i < scores.Rows; i++ {
		for j := 0; j < scores.Cols; j++ {
			scores.Data[i][j] /= scaleFactor
		}
	}
	
	// Apply mask if provided
	if mask != nil {
		scores = mask.ApplyMask(scores)
	}
	
	// Apply softmax
	attentionWeights := Softmax(scores)
	
	// Apply dropout to attention weights if training
	if isTraining {
		attentionWeights = aa.AttentionDropout.Forward(attentionWeights, true)
	}
	
	// Apply attention weights
	output := MatMul(attentionWeights, v)
	
	// Project back to model dimension
	return MatMul(output, aa.OutputWeight)
}
