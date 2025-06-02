package transformer

import (
	"math/rand"
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
func (d *Dropout) Forward(input *Matrix, isTraining bool) *Matrix {
	if !isTraining || d.Rate <= 0.0 {
		return input
	}
	
	// Create dropout mask
	d.Mask = NewMatrix(input.Rows, input.Cols)
	
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
	result := NewMatrix(input.Rows, input.Cols)
	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			result.Data[i][j] = input.Data[i][j] * d.Mask.Data[i][j]
		}
	}
	
	return result
}
