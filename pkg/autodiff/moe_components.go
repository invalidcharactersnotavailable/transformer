package autodiff

import (
	"fmt"
	"math"
)

// Expert struct and methods

// Expert represents a feed-forward network, one of many in an MoE layer.
// This type was moved from pkg/moe/moe_components.go to pkg/autodiff/moe_components.go
// to break an import cycle.
type Expert struct {
	W1         *Tensor
	B1         *Tensor
	W2         *Tensor
	B2         *Tensor
	Activation func(*Tensor) (*Tensor, error)
	Graph      *ComputationGraph
}

// NewExpert creates a new expert.
func NewExpert(inputDim, hiddenDim, outputDim int, activation func(*Tensor) (*Tensor, error), requiresGrad bool, graph *ComputationGraph) *Expert {
	limitW1 := math.Sqrt(6.0 / float64(inputDim+hiddenDim))
	limitW2 := math.Sqrt(6.0 / float64(hiddenDim+outputDim))

	w1Config := &TensorConfig{RequiresGrad: requiresGrad, Name: "expert_w1", Graph: graph}
	b1Config := &TensorConfig{RequiresGrad: requiresGrad, Name: "expert_b1", Graph: graph}
	w2Config := &TensorConfig{RequiresGrad: requiresGrad, Name: "expert_w2", Graph: graph}
	b2Config := &TensorConfig{RequiresGrad: requiresGrad, Name: "expert_b2", Graph: graph}

	w1 := NewUniformTensorFallback(inputDim, hiddenDim, -limitW1, limitW1, w1Config)
	b1 := NewZerosTensorFallback(1, hiddenDim, b1Config)
	w2 := NewUniformTensorFallback(hiddenDim, outputDim, -limitW2, limitW2, w2Config)
	b2 := NewZerosTensorFallback(1, outputDim, b2Config)

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
func (e *Expert) Forward(input *Tensor) (*Tensor, error) {
	var err error
	hiddenLayer, err := MatMul(input, e.W1)
	if err != nil { return nil, fmt.Errorf("expert MatMul W1 failed: %w", err) }

	hiddenLayer, err = Add(hiddenLayer, e.B1)
	if err != nil { return nil, fmt.Errorf("expert Add B1 failed: %w", err) }

	if e.Activation != nil {
		hiddenLayer, err = e.Activation(hiddenLayer)
		if err != nil { return nil, fmt.Errorf("expert activation failed: %w", err) }
	}

	outputLayer, err := MatMul(hiddenLayer, e.W2)
	if err != nil { return nil, fmt.Errorf("expert MatMul W2 failed: %w", err) }

	outputLayer, err = Add(outputLayer, e.B2)
	if err != nil { return nil, fmt.Errorf("expert Add B2 failed: %w", err) }

	return outputLayer, nil
}

// GetParameters returns a slice of tensors that are the expert's parameters.
func (e *Expert) GetParameters() []*Tensor {
	return []*Tensor{e.W1, e.B1, e.W2, e.B2}
}

// Router struct and methods

// Router determines which expert(s) each token should be sent to.
// This type was moved from pkg/moe/moe_components.go to pkg/autodiff/moe_components.go
// to break an import cycle.
type Router struct {
	Weights *Tensor
	Bias    *Tensor
	Graph   *ComputationGraph
}

// NewRouter creates a new router.
func NewRouter(modelDim, numExperts int, requiresGrad bool, graph *ComputationGraph) *Router {
	limitWeights := math.Sqrt(6.0 / float64(modelDim+numExperts))
	weightsConfig := &TensorConfig{RequiresGrad: requiresGrad, Name: "router_weights", Graph: graph}
	biasConfig := &TensorConfig{RequiresGrad: requiresGrad, Name: "router_bias", Graph: graph}

	weights := NewUniformTensorFallback(modelDim, numExperts, -limitWeights, limitWeights, weightsConfig)
	bias := NewZerosTensorFallback(1, numExperts, biasConfig)

	return &Router{
		Weights: weights,
		Bias:    bias,
		Graph:   graph,
	}
}

// Forward computes the routing scores (logits) for each token and each expert.
func (r *Router) Forward(tokens *Tensor) (*Tensor, error) {
	logits, err := MatMul(tokens, r.Weights)
	if err != nil { return nil, fmt.Errorf("router MatMul failed: %w", err) }

	logits, err = Add(logits, r.Bias)
	if err != nil { return nil, fmt.Errorf("router Add Bias failed: %w", err) }

	return logits, nil
}

// GetParameters returns a slice of tensors that are the router's parameters.
func (r *Router) GetParameters() []*Tensor {
	return []*Tensor{r.Weights, r.Bias}
}

// NewUniformTensorFallback provides a fallback if NewUniformTensor is not in autodiff.
func NewUniformTensorFallback(rows, cols int, min, max float64, config *TensorConfig) *Tensor {
	// This function was originally in pkg/moe/moe_components.go and used autodiff.NewRandomTensor.
	// Now that it's in pkg/autodiff, it can directly use NewRandomTensor.
	// The warning might still be relevant if NewRandomTensor isn't a true uniform distribution.
	fmt.Printf("Warning: NewUniformTensorFallback is using NewRandomTensor. Proper uniform initialization is recommended.\n")
	tensor, err := NewRandomTensor(rows, cols, config)
	if err != nil {
		panic(fmt.Sprintf("NewRandomTensor failed in fallback: %v", err))
	}
	return tensor
}

// NewZerosTensorFallback provides a fallback if NewZerosTensor with config is not in autodiff.
func NewZerosTensorFallback(rows, cols int, config *TensorConfig) *Tensor {
	// This function was originally in pkg/moe/moe_components.go and used autodiff.NewZerosTensor.
	// Now that it's in pkg/autodiff, it can directly use NewZerosTensor.
	// The function signature was (rows, cols int, config *TensorConfig)
	// but autodiff.NewZerosTensor is (config *TensorConfig, rows, cols int)
	// Adjusting the call accordingly.
	tensor, err := NewZerosTensor(config, rows, cols)
	if err != nil {
		panic(fmt.Sprintf("NewZerosTensor failed in fallback: %v", err))
	}
	return tensor
}
