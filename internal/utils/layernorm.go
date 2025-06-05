package transformer

import (
	"math"
	"transformer/pkg/autodiff"
)

// LayerNorm represents a layer normalization component
type LayerNorm struct {
	Dim      int
	Epsilon  float64
	Gamma    *autodiff.Matrix
	Beta     *autodiff.Matrix
}

// NewLayerNorm creates a new layer normalization component
func NewLayerNorm(dim int) *LayerNorm {
	gamma := autodiff.MustNewMatrix(1, dim)
	beta := autodiff.MustNewMatrix(1, dim)
	
	// Initialize gamma to ones and beta to zeros
	for i := 0; i < dim; i++ {
		gamma.Data[0][i] = 1.0
		beta.Data[0][i] = 0.0
	}
	
	return &LayerNorm{
		Dim:     dim,
		Epsilon: 1e-5,
		Gamma:   gamma,
		Beta:    beta,
	}
}

// Forward applies layer normalization to the input
func (ln *LayerNorm) Forward(input *autodiff.Matrix) *autodiff.Matrix {
	output := autodiff.MustNewMatrix(input.Rows, input.Cols)
	
	for i := 0; i < input.Rows; i++ {
		// Calculate mean
		mean := 0.0
		for j := 0; j < input.Cols; j++ {
			mean += input.Data[i][j]
		}
		mean /= float64(input.Cols)
		
		// Calculate variance
		variance := 0.0
		for j := 0; j < input.Cols; j++ {
			diff := input.Data[i][j] - mean
			variance += diff * diff
		}
		variance /= float64(input.Cols)
		
		// Normalize
		for j := 0; j < input.Cols; j++ {
			normalized := (input.Data[i][j] - mean) / math.Sqrt(variance+ln.Epsilon)
			output.Data[i][j] = normalized*ln.Gamma.Data[0][j] + ln.Beta.Data[0][j]
		}
	}
	
	return output
}
