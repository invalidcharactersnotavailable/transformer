package transformer

import (
	"math"
)

// FeedForwardType represents the type of activation function to use
type FeedForwardType int

const (
	// ReLUActivation uses ReLU activation function
	ReLUActivation FeedForwardType = iota
	// GELUActivation uses GELU activation function
	GELUActivation
)

// FeedForward represents a configurable feed-forward neural network
// that supports different activation functions and optional dropout
type FeedForward struct {
	InputDim     int
	HiddenDim    int
	W1           *Matrix
	B1           *Matrix
	W2           *Matrix
	B2           *Matrix
	Dropout      *Dropout
	DropoutRate  float64
	ActivationType FeedForwardType
}

// FeedForwardConfig holds configuration options for creating a FeedForward network
type FeedForwardConfig struct {
	InputDim      int
	HiddenDim     int
	DropoutRate   float64
	ActivationType FeedForwardType
}

// NewFeedForward creates a new feed-forward network with the specified configuration
func NewFeedForward(config FeedForwardConfig) (*FeedForward, error) {
	if config.InputDim <= 0 {
		return nil, fmt.Errorf("input dimension must be positive, got %d", config.InputDim)
	}
	
	if config.HiddenDim <= 0 {
		return nil, fmt.Errorf("hidden dimension must be positive, got %d", config.HiddenDim)
	}
	
	if config.DropoutRate < 0 || config.DropoutRate >= 1.0 {
		return nil, fmt.Errorf("dropout rate must be in range [0, 1), got %f", config.DropoutRate)
	}
	
	return &FeedForward{
		InputDim:      config.InputDim,
		HiddenDim:     config.HiddenDim,
		W1:            NewRandomMatrix(config.InputDim, config.HiddenDim),
		B1:            NewMatrix(1, config.HiddenDim),
		W2:            NewRandomMatrix(config.HiddenDim, config.InputDim),
		B2:            NewMatrix(1, config.InputDim),
		Dropout:       NewDropout(config.DropoutRate),
		DropoutRate:   config.DropoutRate,
		ActivationType: config.ActivationType,
	}, nil
}

// NewDefaultFeedForward creates a new feed-forward network with ReLU activation and no dropout
func NewDefaultFeedForward(inputDim, hiddenDim int) (*FeedForward, error) {
	return NewFeedForward(FeedForwardConfig{
		InputDim:      inputDim,
		HiddenDim:     hiddenDim,
		DropoutRate:   0.0,
		ActivationType: ReLUActivation,
	})
}

// NewAdvancedFeedForward creates a new feed-forward network with GELU activation and dropout
func NewAdvancedFeedForward(inputDim, hiddenDim int, dropoutRate float64) (*FeedForward, error) {
	return NewFeedForward(FeedForwardConfig{
		InputDim:      inputDim,
		HiddenDim:     hiddenDim,
		DropoutRate:   dropoutRate,
		ActivationType: GELUActivation,
	})
}

// applyReLU applies the ReLU activation function to a matrix
func applyReLU(m *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if m.Data[i][j] < 0 {
				m.Data[i][j] = 0
			}
		}
	}
}

// applyGELU applies the GELU activation function to a matrix
func applyGELU(m *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] = gelu(m.Data[i][j])
		}
	}
}

// gelu calculates the Gaussian Error Linear Unit activation
func gelu(x float64) float64 {
	// Approximation of GELU
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3.0))))
}

// addBias adds bias values to each row of a matrix
func addBias(m *Matrix, bias *Matrix) {
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			m.Data[i][j] += bias.Data[0][j]
		}
	}
}

// Forward performs the feed-forward operation
// If isTraining is true, dropout will be applied (if configured)
func (ff *FeedForward) Forward(input *Matrix, isTraining bool) (*Matrix, error) {
	if input == nil {
		return nil, fmt.Errorf("input matrix cannot be nil")
	}
	
	if input.Cols != ff.InputDim {
		return nil, fmt.Errorf("input dimension mismatch: expected %d, got %d", ff.InputDim, input.Cols)
	}
	
	// First linear transformation
	hidden := MatMul(input, ff.W1)
	if hidden == nil {
		return nil, fmt.Errorf("matrix multiplication failed")
	}
	
	// Add bias
	addBias(hidden, ff.B1)
	
	// Apply activation function
	switch ff.ActivationType {
	case ReLUActivation:
		applyReLU(hidden)
	case GELUActivation:
		applyGELU(hidden)
	default:
		return nil, fmt.Errorf("unknown activation type: %v", ff.ActivationType)
	}
	
	// Apply dropout if training and dropout is configured
	if isTraining && ff.DropoutRate > 0 {
		hidden = ff.Dropout.Forward(hidden, true)
		if hidden == nil {
			return nil, fmt.Errorf("dropout operation failed")
		}
	}
	
	// Second linear transformation
	output := MatMul(hidden, ff.W2)
	if output == nil {
		return nil, fmt.Errorf("matrix multiplication failed")
	}
	
	// Add bias
	addBias(output, ff.B2)
	
	return output, nil
}

// ForwardLegacy provides backward compatibility with the original API
// that doesn't require error handling
func (ff *FeedForward) ForwardLegacy(input *Matrix) *Matrix {
	output, err := ff.Forward(input, false)
	if err != nil {
		// In legacy mode, we'll return a zero matrix on error
		return NewMatrix(input.Rows, ff.InputDim)
	}
	return output
}

// Clone creates a deep copy of the FeedForward network
func (ff *FeedForward) Clone() *FeedForward {
	return &FeedForward{
		InputDim:      ff.InputDim,
		HiddenDim:     ff.HiddenDim,
		W1:            ff.W1.Clone(),
		B1:            ff.B1.Clone(),
		W2:            ff.W2.Clone(),
		B2:            ff.B2.Clone(),
		Dropout:       ff.Dropout.Clone(),
		DropoutRate:   ff.DropoutRate,
		ActivationType: ff.ActivationType,
	}
}
