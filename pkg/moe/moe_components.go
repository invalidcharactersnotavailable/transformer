package moe

import (
	"math"
	"github.com/transformer_reorganized/pkg/autodiff"
)

// Expert struct and methods

// Expert represents a feed-forward network, one of many in an MoE layer.
type Expert struct {
	W1         *autodiff.Tensor
	B1         *autodiff.Tensor
	W2         *autodiff.Tensor
	B2         *autodiff.Tensor
	Activation func(*autodiff.Tensor) (*autodiff.Tensor, error) // Now returns error
	Graph      *autodiff.ComputationGraph // So expert ops can be part of the graph
}

// NewExpert creates a new expert.
// inputDim: Dimension of the input tensor.
// hiddenDim: Dimension of the hidden layer.
// outputDim: Dimension of the output tensor.
// activation: The activation function to use (e.g., autodiff.TensorGELU).
// requiresGrad: Whether the expert's parameters require gradients.
// graph: The computation graph.
func NewExpert(inputDim, hiddenDim, outputDim int, activation func(*autodiff.Tensor) (*autodiff.Tensor, error), requiresGrad bool, graph *autodiff.ComputationGraph) *Expert {
	// Initialize weights using Glorot/Xavier uniform initialization.
	limitW1 := math.Sqrt(6.0 / float64(inputDim+hiddenDim))
	limitW2 := math.Sqrt(6.0 / float64(hiddenDim+outputDim))

	w1Config := &autodiff.TensorConfig{RequiresGrad: requiresGrad, Name: "expert_w1", Graph: graph}
	b1Config := &autodiff.TensorConfig{RequiresGrad: requiresGrad, Name: "expert_b1", Graph: graph}
	w2Config := &autodiff.TensorConfig{RequiresGrad: requiresGrad, Name: "expert_w2", Graph: graph}
	b2Config := &autodiff.TensorConfig{RequiresGrad: requiresGrad, Name: "expert_b2", Graph: graph}

	w1 := autodiff.NewUniformTensorFallback(inputDim, hiddenDim, -limitW1, limitW1, w1Config) // Use fallback if NewUniformTensor not present
	b1 := autodiff.NewZerosTensorFallback(1, hiddenDim, b1Config)
	w2 := autodiff.NewUniformTensorFallback(hiddenDim, outputDim, -limitW2, limitW2, w2Config)
	b2 := autodiff.NewZerosTensorFallback(1, outputDim, b2Config)

	return &Expert{
		W1:         w1,
		B1:         b1,
		W2:         w2,
		B2:         b2,
		Activation: activation,
		Graph:      graph,
	}
}

// Forward processes the input tensor through the expert.
// Input shape: (num_tokens_for_expert, inputDim)
// Output shape: (num_tokens_for_expert, outputDim)
func (e *Expert) Forward(input *autodiff.Tensor) (*autodiff.Tensor, error) {
	var err error
	// Hidden layer: h = activation(input @ W1 + B1)
	hiddenLayer, err := autodiff.MatMul(input, e.W1)
	if err != nil { return nil, fmt.Errorf("expert MatMul W1 failed: %w", err) }

	hiddenLayer, err = autodiff.Add(hiddenLayer, e.B1) // B1 is broadcasted
	if err != nil { return nil, fmt.Errorf("expert Add B1 failed: %w", err) }

	if e.Activation != nil {
		hiddenLayer, err = e.Activation(hiddenLayer)
		if err != nil { return nil, fmt.Errorf("expert activation failed: %w", err) }
	}

	// Output layer: output = h @ W2 + B2
	outputLayer, err := autodiff.MatMul(hiddenLayer, e.W2)
	if err != nil { return nil, fmt.Errorf("expert MatMul W2 failed: %w", err) }

	outputLayer, err = autodiff.Add(outputLayer, e.B2) // B2 is broadcasted
	if err != nil { return nil, fmt.Errorf("expert Add B2 failed: %w", err) }

	return outputLayer, nil
}

// GetParameters returns a slice of tensors that are the expert's parameters.
func (e *Expert) GetParameters() []*autodiff.Tensor {
	return []*autodiff.Tensor{e.W1, e.B1, e.W2, e.B2}
}

// Router struct and methods

// Router determines which expert(s) each token should be sent to.
type Router struct {
	Weights *autodiff.Tensor // modelDim x numExperts
	Bias    *autodiff.Tensor // 1 x numExperts
	Graph   *autodiff.ComputationGraph
}

// NewRouter creates a new router.
func NewRouter(modelDim, numExperts int, requiresGrad bool, graph *autodiff.ComputationGraph) *Router {
	limitWeights := math.Sqrt(6.0 / float64(modelDim+numExperts))
	weightsConfig := &autodiff.TensorConfig{RequiresGrad: requiresGrad, Name: "router_weights", Graph: graph}
	biasConfig := &autodiff.TensorConfig{RequiresGrad: requiresGrad, Name: "router_bias", Graph: graph}

	weights := autodiff.NewUniformTensorFallback(modelDim, numExperts, -limitWeights, limitWeights, weightsConfig)
	bias := autodiff.NewZerosTensorFallback(1, numExperts, biasConfig)

	return &Router{
		Weights: weights,
		Bias:    bias,
		Graph:   graph,
	}
}

// Forward computes the routing scores (logits) for each token and each expert.
// tokens shape: (batch_seq_len, modelDim)
// Output logits shape: (batch_seq_len, numExperts)
func (r *Router) Forward(tokens *autodiff.Tensor) (*autodiff.Tensor, error) { // Added error return
	logits, err := autodiff.MatMul(tokens, r.Weights)
	if err != nil { return nil, fmt.Errorf("router MatMul failed: %w", err) }

	logits, err = autodiff.Add(logits, r.Bias) // Bias is broadcasted
	if err != nil { return nil, fmt.Errorf("router Add Bias failed: %w", err) }

	return logits, nil
}

// GetParameters returns a slice of tensors that are the router's parameters.
func (r *Router) GetParameters() []*autodiff.Tensor {
	return []*autodiff.Tensor{r.Weights, r.Bias}
}

// NewUniformTensorFallback provides a fallback if NewUniformTensor is not in autodiff.
// It uses NewRandomTensor as a substitute.
func ( /* *autodiff.Tensor -- receiver not needed for static helper */ ) NewUniformTensorFallback(rows, cols int, min, max float64, config *autodiff.TensorConfig) *autodiff.Tensor {
	// In a real scenario, NewUniformTensor would be implemented in autodiff package.
	// For now, using NewRandomTensor which initializes with small random values.
	// This is just to make the code runnable. Proper initialization is important.
	fmt.Printf("Warning: NewUniformTensorFallback is using NewRandomTensor. Proper uniform initialization is recommended.\n")
	tensor, err := autodiff.NewRandomTensor(rows, cols, config)
	if err != nil {
		panic(fmt.Sprintf("NewRandomTensor failed in fallback: %v", err)) // Panic for critical test setup issue
	}
	// One could manually scale/shift NewRandomTensor's output if its range is known,
	// but that's an approximation.
	return tensor
}

// NewZerosTensorFallback provides a fallback if NewZerosTensor with config is not in autodiff.
func ( /* *autodiff.Tensor */ ) NewZerosTensorFallback(rows, cols int, config *autodiff.TensorConfig) *autodiff.Tensor {
	tensor, err := autodiff.NewZerosTensor(rows, cols, config) // Assuming NewZerosTensor exists and takes config
	if err != nil {
		panic(fmt.Sprintf("NewZerosTensor failed in fallback: %v", err))
	}
	return tensor
}
