package utils

import (
	"math/rand"
	"fmt" // Added fmt import
)

// Dropout represents a dropout layer for regularization
type Dropout struct {
	Rate float64
	Mask *Matrix
}

// NewDropout creates a new dropout layer with specified dropout rate
func NewDropout(rate float64) *Dropout {
	return &Dropout{
		Rate: rate,
	}
}

// Forward applies dropout to the input during training
func (d *Dropout) Forward(input *Matrix, isTraining bool) (*Matrix, error) { // Added error return
	if !isTraining || d.Rate <= 0.0 {
		return input, nil // Return input and nil error
	}
	
	// Create dropout mask
	var err error
	d.Mask, err = NewMatrix(input.Rows, input.Cols) // Handle error
	if err != nil {
		return nil, fmt.Errorf("failed to create mask matrix in dropout: %w", err)
	}
	
	// Scale factor to maintain expected value
	scale := 1.0 / (1.0 - d.Rate)
	
	// Generate binary mask with scaling
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			if rand.Float64() > d.Rate {
				d.Mask.Data[i][j] = scale
			} else {
				d.Mask.Data[i][j] = 0.0
			}
		}
	}
	
	// Apply mask
	result, err := NewMatrix(input.Rows, input.Cols) // Handle error
	if err != nil {
		return nil, fmt.Errorf("failed to create result matrix in dropout: %w", err)
	}
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			result.Data[i][j] = input.Data[i][j] * d.Mask.Data[i][j]
		}
	}
	
	return result, nil // Return result and nil error
}

// Clone creates a new Dropout layer with the same rate.
// Note: The mask is not copied as it's stateful per forward pass.
func (d *Dropout) Clone() *Dropout {
	return NewDropout(d.Rate)
}
