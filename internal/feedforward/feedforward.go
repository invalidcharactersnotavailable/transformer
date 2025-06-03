package transformer

// FeedForward represents a simple feed-forward neural network
type FeedForward struct {
	InputDim  int
	HiddenDim int
	W1        *Matrix
	B1        *Matrix
	W2        *Matrix
	B2        *Matrix
}

// NewFeedForward creates a new feed-forward network
func NewFeedForward(inputDim, hiddenDim int) *FeedForward {
	return &FeedForward{
		InputDim:  inputDim,
		HiddenDim: hiddenDim,
		W1:        NewRandomMatrix(inputDim, hiddenDim),
		B1:        NewMatrix(1, hiddenDim),
		W2:        NewRandomMatrix(hiddenDim, inputDim),
		B2:        NewMatrix(1, inputDim),
	}
}

// Forward performs the feed-forward operation
func (ff *FeedForward) Forward(input *Matrix) *Matrix {
	// First linear transformation
	hidden := MatMul(input, ff.W1)
	
	// Add bias
	for i := 0; i < hidden.Rows; i++ {
		for j := 0; j < hidden.Cols; j++ {
			hidden.Data[i][j] += ff.B1.Data[0][j]
		}
	}
	
	// Apply ReLU activation
	for i := 0; i < hidden.Rows; i++ {
		for j := 0; j < hidden.Cols; j++ {
			if hidden.Data[i][j] < 0 {
				hidden.Data[i][j] = 0
			}
		}
	}
	
	// Second linear transformation
	output := MatMul(hidden, ff.W2)
	
	// Add bias
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			output.Data[i][j] += ff.B2.Data[0][j]
		}
	}
	
	return output
}
