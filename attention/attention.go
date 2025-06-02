package transformer

import (
	"math"
)

// MultiHeadAttention represents a multi-head attention mechanism
type MultiHeadAttention struct {
	NumHeads    int
	ModelDim    int
	HeadDim     int
	QueryWeight *Matrix
	KeyWeight   *Matrix
	ValueWeight *Matrix
	OutputWeight *Matrix
}

// NewMultiHeadAttention creates a new multi-head attention layer
func NewMultiHeadAttention(numHeads, modelDim int) *MultiHeadAttention {
	headDim := modelDim / numHeads
	
	return &MultiHeadAttention{
		NumHeads:    numHeads,
		ModelDim:    modelDim,
		HeadDim:     headDim,
		QueryWeight: NewRandomMatrix(modelDim, modelDim),
		KeyWeight:   NewRandomMatrix(modelDim, modelDim),
		ValueWeight: NewRandomMatrix(modelDim, modelDim),
		OutputWeight: NewRandomMatrix(modelDim, modelDim),
	}
}

// Forward performs the multi-head attention operation
func (mha *MultiHeadAttention) Forward(query, key, value *Matrix) *Matrix {
	// Project inputs to query, key, value
	q := MatMul(query, mha.QueryWeight)
	k := MatMul(key, mha.KeyWeight)
	v := MatMul(value, mha.ValueWeight)
	
	// Simple scaled dot-product attention
	// For simplicity, we're not splitting into multiple heads in this minimal implementation
	kT := Transpose(k)
	scores := MatMul(q, kT)
	
	// Scale
	scaleFactor := math.Sqrt(float64(mha.ModelDim))
	for i := 0; i < scores.Rows; i++ {
		for j := 0; j < scores.Cols; j++ {
			scores.Data[i][j] /= scaleFactor
		}
	}
	
	// Apply softmax
	attentionWeights := Softmax(scores)
	
	// Apply attention weights
	output := MatMul(attentionWeights, v)
	
	// Project back to model dimension
	return MatMul(output, mha.OutputWeight)
}
