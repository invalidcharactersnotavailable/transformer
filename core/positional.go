package transformer

import (
	"math"
)

// PositionalEncoding represents positional encoding for transformer inputs
type PositionalEncoding struct {
	Dim      int
	MaxLen   int
	Encoding *Matrix
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

// AddToEmbedding adds positional encoding to the input embeddings
func (pe *PositionalEncoding) AddToEmbedding(embeddings *Matrix) *Matrix {
	if embeddings.Rows > pe.MaxLen {
		panic("Input sequence length exceeds maximum length for positional encoding")
	}
	
	result := NewMatrix(embeddings.Rows, embeddings.Cols)
	
	for i := 0; i < embeddings.Rows; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			result.Data[i][j] = embeddings.Data[i][j] + pe.Encoding.Data[i][j]
		}
	}
	
	return result
}
