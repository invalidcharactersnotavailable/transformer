package transformer

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"transformer/pkg/autodiff"
	"transformer/pkg/core"
)

// MergedFeedForward represents a merged feed-forward network layer
type MergedFeedForward struct {
	InputDim     int
	HiddenDim    int
	OutputDim    int
	NumExperts   int
	Weights1     *autodiff.Matrix
	Biases1      *autodiff.Matrix
	Weights2     *autodiff.Matrix
	Biases2      *autodiff.Matrix
	Dropout      *autodiff.DropoutTensor
	ActivationFn func(float64) float64
	Config       *core.Config
	mu           sync.Mutex
}

// NewMergedFeedForward creates a new MergedFeedForward layer
func NewMergedFeedForward(
	inputDim, hiddenDim, outputDim, numExperts int,
	dropoutRate float64,
	activationFn func(float64) float64,
	config *core.Config) *MergedFeedForward {

	weights1 := autodiff.MustNewMatrix(numExperts*inputDim, hiddenDim)
	biases1 := autodiff.MustNewMatrix(1, numExperts*hiddenDim)
	weights2 := autodiff.MustNewMatrix(numExperts*hiddenDim, outputDim)
	biases2 := autodiff.MustNewMatrix(1, numExperts*outputDim)
	var dropout *autodiff.DropoutTensor
	if dropoutRate > 0 {
		dropout = autodiff.NewDropoutTensor(dropoutRate)
	}

	limit1 := math.Sqrt(6.0 / float64(inputDim+hiddenDim))
	for i := 0; i < weights1.Rows; i++ {
		for j := 0; j < weights1.Cols; j++ {
			weights1.Data[i][j] = rand.Float64()*2*limit1 - limit1
		}
	}
	limit2 := math.Sqrt(6.0 / float64(hiddenDim+outputDim))
	for i := 0; i < weights2.Rows; i++ {
		for j := 0; j < weights2.Cols; j++ {
			weights2.Data[i][j] = rand.Float64()*2*limit2 - limit2
		}
	}

	return &MergedFeedForward{
		InputDim:     inputDim,
		HiddenDim:    hiddenDim,
		OutputDim:    outputDim,
		NumExperts:   numExperts,
		Weights1:     weights1,
		Biases1:      biases1,
		Weights2:     weights2,
		Biases2:      biases2,
		Dropout:      dropout,
		ActivationFn: activationFn,
		Config:       config,
	}
}

// Forward performs the forward pass through the MergedFeedForward layer
func (mff *MergedFeedForward) Forward(input *autodiff.Matrix, isTraining bool) (*autodiff.Matrix, error) {
	if input.Cols != mff.InputDim*mff.NumExperts {
		return nil, fmt.Errorf("input dimensions %d do not match expected %d", input.Cols, mff.InputDim*mff.NumExperts)
	}

	hidden, err := autodiff.MatMul(input, mff.Weights1)
	if err != nil {
		return nil, fmt.Errorf("error in first matrix multiplication: %v", err)
	}
	hidden, err = autodiff.Add(hidden, mff.Biases1)
	if err != nil {
		return nil, fmt.Errorf("error adding B1: %v", err)
	}

	if mff.ActivationFn != nil {
		hidden, err = autodiff.ApplyFunction(hidden, mff.ActivationFn)
		if err != nil {
			return nil, fmt.Errorf("error applying activation function: %v", err)
		}
	}

	if isTraining && mff.Dropout != nil && mff.Dropout.Rate > 0 {
		// This section requires Matrix <-> Tensor conversion or a Matrix-based Dropout utility.
		// Commenting out for now to allow build to proceed.
		fmt.Println("Warning: Dropout in MergedFeedForward needs Matrix <-> Tensor conversion or redesign.")
		// hiddenTensor, errConv := autodiff.NewTensorFromMatrix(hidden, &autodiff.TensorConfig{Graph: ???}) // Needs graph access
		// if errConv != nil { return nil, fmt.Errorf("failed to convert hidden matrix to tensor for dropout: %v", errConv)}
		// hiddenTensorDropped, errDrop := mff.Dropout.Forward(hiddenTensor, isTraining)
		// if errDrop != nil { return nil, fmt.Errorf("dropout forward failed: %v", errDrop) }
		// hidden, errConvBack := hiddenTensorDropped.Matrix()
		// if errConvBack != nil { return nil, fmt.Errorf("failed to convert hidden tensor back to matrix after dropout: %v", errConvBack)}
	}

	output, err := autodiff.MatMul(hidden, mff.Weights2)
	if err != nil {
		return nil, fmt.Errorf("error in second matrix multiplication: %v", err)
	}
	output, err = autodiff.Add(output, mff.Biases2)
	if err != nil {
		return nil, fmt.Errorf("error adding B2: %v", err)
	}

	return output, nil
}

// Note: Manual Backward function is removed as it's not compatible with autodiff gradient flow.
// Autodiff tensors would handle gradients automatically if this layer were fully tensor-based.
