package transformer

// AdvancedFeedForward represents an enhanced feed-forward neural network with GELU activation
type AdvancedFeedForward struct {
	InputDim  int
	HiddenDim int
	W1        *Matrix
	B1        *Matrix
	W2        *Matrix
	B2        *Matrix
	Dropout   *Dropout
}

// NewAdvancedFeedForward creates a new advanced feed-forward network
func NewAdvancedFeedForward(inputDim, hiddenDim int, dropoutRate float64) *AdvancedFeedForward {
	return &AdvancedFeedForward{
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
		W1:        NewRandomMatrix(inputDim, hiddenDim),
		B1:        NewMatrix(1, hiddenDim),
		W2:        NewRandomMatrix(hiddenDim, inputDim),
		B2:        NewMatrix(1, inputDim),
		Dropout:   NewDropout(dropoutRate),
	}
}

// GELU activation function (Gaussian Error Linear Unit)
func GELU(x float64) float64 {
	// Approximation of GELU
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3.0))))
}

// Forward performs the advanced feed-forward operation with GELU activation
func (aff *AdvancedFeedForward) Forward(input *Matrix, isTraining bool) *Matrix {
	// First linear transformation
	hidden := MatMul(input, aff.W1)
	
	// Add bias
	for i := 0; i < hidden.Rows; i++ {
		for j := 0; j < hidden.Cols; j++ {
			hidden.Data[i][j] += aff.B1.Data[0][j]
		}
	}
	
	// Apply GELU activation
	for i := 0; i < hidden.Rows; i++ {
		for j := 0; j < hidden.Cols; j++ {
			hidden.Data[i][j] = GELU(hidden.Data[i][j])
		}
	}
	
	// Apply dropout if training
	if isTraining {
		hidden = aff.Dropout.Forward(hidden, true)
	}
	
	// Second linear transformation
	output := MatMul(hidden, aff.W2)
	
	// Add bias
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			output.Data[i][j] += aff.B2.Data[0][j]
		}
	}
	
	return output
}
