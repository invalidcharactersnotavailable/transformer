package utils

import (
	"math"
	"fmt" // Added fmt import
)

// LayerNorm represents a layer normalization component
type LayerNorm struct {
	Dim      int
	Epsilon  float64
	Gamma    *Matrix
	Beta     *Matrix
}

// NewLayerNorm creates a new layer normalization component
func NewLayerNorm(dim int) (*LayerNorm, error) { // Added error return
	gamma, err := NewMatrix(1, dim) // Handle error
	if err != nil {
		return nil, fmt.Errorf("failed to create gamma matrix in layernorm: %w", err)
	}
	beta, err := NewMatrix(1, dim) // Handle error
	if err != nil {
		return nil, fmt.Errorf("failed to create beta matrix in layernorm: %w", err)
	}
	
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
	}, nil // Return nil error
}

// Forward applies layer normalization to the input
func (ln *LayerNorm) Forward(input *Matrix) (*Matrix, error) { // Added error return
	output, err := NewMatrix(input.Rows, input.Cols) // Handle error
	if err != nil {
		return nil, fmt.Errorf("failed to create output matrix in layernorm forward: %w", err)
	}
	
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
	
	return output, nil // Return nil error
}
