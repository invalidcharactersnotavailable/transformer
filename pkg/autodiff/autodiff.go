package autodiff

import (
	"fmt"
	"math"
)

// Tensor represents a tensor with gradient tracking capabilities
type Tensor struct {
	Data       *Matrix
	Grad       *Matrix
	Requires   bool
	BackwardFn func() []*Tensor
	Children   []*Tensor
	Name       string // Optional name for debugging
}

// TensorConfig holds configuration options for creating a tensor
type TensorConfig struct {
	RequiresGrad bool
	Name         string
}

// DefaultTensorConfig returns the default configuration for tensors
func DefaultTensorConfig() *TensorConfig {
	return &TensorConfig{
		RequiresGrad: false,
		Name:         "",
	}
}

// NewTensor creates a new tensor from a matrix with the specified configuration
func NewTensor(data *Matrix, config *TensorConfig) (*Tensor, error) {
	if data == nil {
		return nil, fmt.Errorf("data matrix cannot be nil")
	}
	
	if config == nil {
		config = DefaultTensorConfig()
	}
	
	var grad *Matrix
	var err error
	
	if config.RequiresGrad {
		grad, err = NewMatrix(data.Rows, data.Cols)
		if err != nil {
			return nil, fmt.Errorf("failed to create gradient matrix: %v", err)
		}
	}
	
	return &Tensor{
		Data:       data,
		Grad:       grad,
		Requires:   config.RequiresGrad,
		BackwardFn: nil,
		Children:   make([]*Tensor, 0),
		Name:       config.Name,
	}, nil
}

// NewRandomTensor creates a new tensor with random values
func NewRandomTensor(rows, cols int, config *TensorConfig) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("dimensions must be positive: rows=%d, cols=%d", rows, cols)
	}
	
	data, err := NewRandomMatrix(rows, cols)
	if err != nil {
		return nil, fmt.Errorf("failed to create random matrix: %v", err)
	}
	
	return NewTensor(data, config)
}

// NewZerosTensor creates a new tensor filled with zeros
func NewZerosTensor(rows, cols int, config *TensorConfig) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("dimensions must be positive: rows=%d, cols=%d", rows, cols)
	}
	
	data, err := NewMatrix(rows, cols)
	if err != nil {
		return nil, fmt.Errorf("failed to create zero matrix: %v", err)
	}
	
	return NewTensor(data, config)
}

// NewOnesTensor creates a new tensor filled with ones
func NewOnesTensor(rows, cols int, config *TensorConfig) (*Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("dimensions must be positive: rows=%d, cols=%d", rows, cols)
	}
	
	data, err := NewMatrix(rows, cols)
	if err != nil {
		return nil, fmt.Errorf("failed to create ones matrix: %v", err)
	}
	
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data.Data[i][j] = 1.0
		}
	}
	
	return NewTensor(data, config)
}

// ZeroGrad zeros out the gradient
func (t *Tensor) ZeroGrad() error {
	if !t.Requires {
		return fmt.Errorf("cannot zero gradient for tensor that doesn't require gradients")
	}
	
	if t.Grad == nil {
		return fmt.Errorf("gradient matrix is nil")
	}
	
	for i := 0; i < t.Grad.Rows; i++ {
		for j := 0; j < t.Grad.Cols; j++ {
			t.Grad.Data[i][j] = 0.0
		}
	}
	
	return nil
}

// Backward computes gradients
func (t *Tensor) Backward() error {
	// Initialize gradient to 1.0 if it's a scalar
	if t.Data.Rows == 1 && t.Data.Cols == 1 {
		if t.Grad == nil {
			return fmt.Errorf("gradient matrix is nil")
		}
		t.Grad.Data[0][0] = 1.0
	}
	
	// Topological sort for backward pass
	visited := make(map[*Tensor]bool)
	topo := make([]*Tensor, 0)
	
	var buildTopo func(node *Tensor) error
	buildTopo = func(node *Tensor) error {
		if node == nil {
			return fmt.Errorf("cannot build topology for nil tensor")
		}
		
		if visited[node] {
			return nil
		}
		
		visited[node] = true
		
		for _, child := range node.Children {
			if child == nil {
				return fmt.Errorf("nil child in tensor %s", node.Name)
			}
			
			if err := buildTopo(child); err != nil {
				return err
			}
		}
		
		topo = append(topo, node)
		return nil
	}
	
	if err := buildTopo(t); err != nil {
		return fmt.Errorf("failed to build topology: %v", err)
	}
	
	// Backward pass
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		
		if node.BackwardFn != nil {
			parents := node.BackwardFn()
			
			for _, parent := range parents {
				if parent == nil {
					return fmt.Errorf("nil parent returned from backward function of tensor %s", node.Name)
				}
				
				if parent.Requires {
					if parent.Grad == nil || node.Grad == nil {
						return fmt.Errorf("gradient matrix is nil during backward pass")
					}
					
					// Gradient accumulation
					for r := 0; r < parent.Grad.Rows; r++ {
						for c := 0; c < parent.Grad.Cols; c++ {
							parent.Grad.Data[r][c] += node.Grad.Data[r][c]
						}
					}
				}
			}
		}
	}
	
	return nil
}

// MatMul performs matrix multiplication with gradient tracking
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	
	if a.Data.Cols != b.Data.Rows {
		return nil, fmt.Errorf("matrix dimensions don't match for multiplication: a(%dx%d), b(%dx%d)",
			a.Data.Rows, a.Data.Cols, b.Data.Rows, b.Data.Cols)
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires || b.Requires,
		Name:         "matmul_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, b.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < b.Data.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Data.Cols; k++ {
				sum += a.Data.Data[i][k] * b.Data.Data[k][j]
			}
			result.Data.Data[i][j] = sum
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires || b.Requires {
		result.Children = append(result.Children, a, b)
		result.BackwardFn = func() []*Tensor {
			if a.Requires {
				// dL/dA = dL/dC * B^T
				bT, err := Transpose(b.Data)
				if err != nil {
					// In a production environment, we would log this error
					return []*Tensor{a, b}
				}
				
				dA, err := MatMul(result.Grad, bT)
				if err != nil {
					// In a production environment, we would log this error
					return []*Tensor{a, b}
				}
				
				for i := 0; i < dA.Rows; i++ {
					for j := 0; j < dA.Cols; j++ {
						a.Grad.Data[i][j] += dA.Data[i][j]
					}
				}
			}
			
			if b.Requires {
				// dL/dB = A^T * dL/dC
				aT, err := Transpose(a.Data)
				if err != nil {
					// In a production environment, we would log this error
					return []*Tensor{a, b}
				}
				
				dB, err := MatMul(aT, result.Grad)
				if err != nil {
					// In a production environment, we would log this error
					return []*Tensor{a, b}
				}
				
				for i := 0; i < dB.Rows; i++ {
					for j := 0; j < dB.Cols; j++ {
						b.Grad.Data[i][j] += dB.Data[i][j]
					}
				}
			}
			
			return []*Tensor{a, b}
		}
	}
	
	return result, nil
}

// Add performs element-wise addition with gradient tracking
func Add(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	
	if a.Data.Rows != b.Data.Rows || a.Data.Cols != b.Data.Cols {
		return nil, fmt.Errorf("matrix dimensions don't match for addition: a(%dx%d), b(%dx%d)",
			a.Data.Rows, a.Data.Cols, b.Data.Rows, b.Data.Cols)
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires || b.Requires,
		Name:         "add_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			result.Data.Data[i][j] = a.Data.Data[i][j] + b.Data.Data[i][j]
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires || b.Requires {
		result.Children = append(result.Children, a, b)
		result.BackwardFn = func() []*Tensor {
			if a.Requires {
				for i := 0; i < a.Data.Rows; i++ {
					for j := 0; j < a.Data.Cols; j++ {
						a.Grad.Data[i][j] += result.Grad.Data[i][j]
					}
				}
			}
			
			if b.Requires {
				for i := 0; i < b.Data.Rows; i++ {
					for j := 0; j < b.Data.Cols; j++ {
						b.Grad.Data[i][j] += result.Grad.Data[i][j]
					}
				}
			}
			
			return []*Tensor{a, b}
		}
	}
	
	return result, nil
}

// Subtract performs element-wise subtraction with gradient tracking
func Subtract(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	
	if a.Data.Rows != b.Data.Rows || a.Data.Cols != b.Data.Cols {
		return nil, fmt.Errorf("matrix dimensions don't match for subtraction: a(%dx%d), b(%dx%d)",
			a.Data.Rows, a.Data.Cols, b.Data.Rows, b.Data.Cols)
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires || b.Requires,
		Name:         "subtract_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			result.Data.Data[i][j] = a.Data.Data[i][j] - b.Data.Data[i][j]
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires || b.Requires {
		result.Children = append(result.Children, a, b)
		result.BackwardFn = func() []*Tensor {
			if a.Requires {
				for i := 0; i < a.Data.Rows; i++ {
					for j := 0; j < a.Data.Cols; j++ {
						a.Grad.Data[i][j] += result.Grad.Data[i][j]
					}
				}
			}
			
			if b.Requires {
				for i := 0; i < b.Data.Rows; i++ {
					for j := 0; j < b.Data.Cols; j++ {
						b.Grad.Data[i][j] -= result.Grad.Data[i][j] // Note the negative sign
					}
				}
			}
			
			return []*Tensor{a, b}
		}
	}
	
	return result, nil
}

// Multiply performs element-wise multiplication (Hadamard product) with gradient tracking
func Multiply(a, b *Tensor) (*Tensor, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("input tensors cannot be nil")
	}
	
	if a.Data.Rows != b.Data.Rows || a.Data.Cols != b.Data.Cols {
		return nil, fmt.Errorf("matrix dimensions don't match for element-wise multiplication: a(%dx%d), b(%dx%d)",
			a.Data.Rows, a.Data.Cols, b.Data.Rows, b.Data.Cols)
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires || b.Requires,
		Name:         "multiply_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			result.Data.Data[i][j] = a.Data.Data[i][j] * b.Data.Data[i][j]
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires || b.Requires {
		result.Children = append(result.Children, a, b)
		result.BackwardFn = func() []*Tensor {
			if a.Requires {
				for i := 0; i < a.Data.Rows; i++ {
					for j := 0; j < a.Data.Cols; j++ {
						a.Grad.Data[i][j] += result.Grad.Data[i][j] * b.Data.Data[i][j]
					}
				}
			}
			
			if b.Requires {
				for i := 0; i < b.Data.Rows; i++ {
					for j := 0; j < b.Data.Cols; j++ {
						b.Grad.Data[i][j] += result.Grad.Data[i][j] * a.Data.Data[i][j]
					}
				}
			}
			
			return []*Tensor{a, b}
		}
	}
	
	return result, nil
}

// ScalarMultiply multiplies a tensor by a scalar value with gradient tracking
func ScalarMultiply(a *Tensor, scalar float64) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "scalar_multiply_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			result.Data.Data[i][j] = a.Data.Data[i][j] * scalar
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					a.Grad.Data[i][j] += result.Grad.Data[i][j] * scalar
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// ReLU applies the ReLU activation function with gradient tracking
func ReLU(a *Tensor) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "relu_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			if a.Data.Data[i][j] > 0 {
				result.Data.Data[i][j] = a.Data.Data[i][j]
			} else {
				result.Data.Data[i][j] = 0
			}
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					if a.Data.Data[i][j] > 0 {
						a.Grad.Data[i][j] += result.Grad.Data[i][j]
					}
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// GELU applies the GELU activation function with gradient tracking
func GELU(a *Tensor) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "gelu_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Constants for GELU approximation
	sqrt2OverPi := math.Sqrt(2.0 / math.Pi)
	coeff := 0.044715
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			x := a.Data.Data[i][j]
			tanh_arg := sqrt2OverPi * (x + coeff*math.Pow(x, 3.0))
			result.Data.Data[i][j] = 0.5 * x * (1.0 + math.Tanh(tanh_arg))
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					x := a.Data.Data[i][j]
					tanh_arg := sqrt2OverPi * (x + coeff*math.Pow(x, 3.0))
					tanh_val := math.Tanh(tanh_arg)
					
					// Derivative of GELU
					dtanh := 1.0 - tanh_val*tanh_val
					inner_deriv := sqrt2OverPi * (1.0 + 3.0*coeff*math.Pow(x, 2.0))
					gelu_grad := 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * inner_deriv
					
					a.Grad.Data[i][j] += result.Grad.Data[i][j] * gelu_grad
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// Softmax applies the softmax function with gradient tracking
func Softmax(a *Tensor) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "softmax_result",
	}
	
	result, err := NewZerosTensor(a.Data.Rows, a.Data.Cols, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		// Find max value in row for numerical stability
		max := a.Data.Data[i][0]
		for j := 1; j < a.Data.Cols; j++ {
			if a.Data.Data[i][j] > max {
				max = a.Data.Data[i][j]
			}
		}
		
		// Calculate sum of exponentials
		sum := 0.0
		for j := 0; j < a.Data.Cols; j++ {
			exp_val := math.Exp(a.Data.Data[i][j] - max)
			result.Data.Data[i][j] = exp_val
			sum += exp_val
		}
		
		// Normalize
		for j := 0; j < a.Data.Cols; j++ {
			result.Data.Data[i][j] /= sum
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					softmax_j := result.Data.Data[i][j]
					for k := 0; k < a.Data.Cols; k++ {
						softmax_k := result.Data.Data[i][k]
						// Jacobian of softmax: J_jk = s_j * (delta_jk - s_k)
						deriv := 0.0
						if j == k {
							deriv = softmax_j * (1.0 - softmax_j)
						} else {
							deriv = -softmax_j * softmax_k
						}
						a.Grad.Data[i][j] += result.Grad.Data[i][k] * deriv
					}
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// CrossEntropyLoss computes the cross-entropy loss with gradient tracking
func CrossEntropyLoss(logits *Tensor, targets []int) (*Tensor, error) {
	if logits == nil {
		return nil, fmt.Errorf("logits tensor cannot be nil")
	}
	
	if targets == nil {
		return nil, fmt.Errorf("targets cannot be nil")
	}
	
	batchSize := logits.Data.Rows
	if len(targets) != batchSize {
		return nil, fmt.Errorf("number of targets (%d) doesn't match batch size (%d)", len(targets), batchSize)
	}
	
	config := &TensorConfig{
		RequiresGrad: logits.Requires,
		Name:         "cross_entropy_loss_result",
	}
	
	result, err := NewZerosTensor(1, 1, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	loss := 0.0
	for i := 0; i < batchSize; i++ {
		if targets[i] < 0 || targets[i] >= logits.Data.Cols {
			return nil, fmt.Errorf("target index out of bounds: %d (must be in [0, %d))", targets[i], logits.Data.Cols)
		}
		
		// -log(softmax) = -log(exp(x_i)/sum(exp(x_j))) = -x_i + log(sum(exp(x_j)))
		
		// Find max for numerical stability
		max := logits.Data.Data[i][0]
		for j := 1; j < logits.Data.Cols; j++ {
			if logits.Data.Data[i][j] > max {
				max = logits.Data.Data[i][j]
			}
		}
		
		// Compute log(sum(exp(x_j - max)))
		sum := 0.0
		for j := 0; j < logits.Data.Cols; j++ {
			sum += math.Exp(logits.Data.Data[i][j] - max)
		}
		logSum := math.Log(sum) + max
		
		// Compute loss
		loss += logSum - logits.Data.Data[i][targets[i]]
	}
	loss /= float64(batchSize)
	result.Data.Data[0][0] = loss
	
	// Set up backward function if gradient is required
	if logits.Requires {
		result.Children = append(result.Children, logits)
		result.BackwardFn = func() []*Tensor {
			// Compute softmax gradients
			for i := 0; i < batchSize; i++ {
				// Find max for numerical stability
				max := logits.Data.Data[i][0]
				for j := 1; j < logits.Data.Cols; j++ {
					if logits.Data.Data[i][j] > max {
						max = logits.Data.Data[i][j]
					}
				}
				
				// Compute softmax
				softmax := make([]float64, logits.Data.Cols)
				sum := 0.0
				for j := 0; j < logits.Data.Cols; j++ {
					softmax[j] = math.Exp(logits.Data.Data[i][j] - max)
					sum += softmax[j]
				}
				for j := 0; j < logits.Data.Cols; j++ {
					softmax[j] /= sum
				}
				
				// Gradient of cross-entropy w.r.t. logits is (softmax - one_hot_target)
				for j := 0; j < logits.Data.Cols; j++ {
					grad := softmax[j]
					if j == targets[i] {
						grad -= 1.0
					}
					logits.Grad.Data[i][j] += grad * result.Grad.Data[0][0] / float64(batchSize)
				}
			}
			return []*Tensor{logits}
		}
	}
	
	return result, nil
}

// MSELoss computes the mean squared error loss with gradient tracking
func MSELoss(predictions *Tensor, targets *Tensor) (*Tensor, error) {
	if predictions == nil || targets == nil {
		return nil, fmt.Errorf("predictions and targets tensors cannot be nil")
	}
	
	if predictions.Data.Rows != targets.Data.Rows || predictions.Data.Cols != targets.Data.Cols {
		return nil, fmt.Errorf("predictions and targets dimensions don't match: predictions(%dx%d), targets(%dx%d)",
			predictions.Data.Rows, predictions.Data.Cols, targets.Data.Rows, targets.Data.Cols)
	}
	
	config := &TensorConfig{
		RequiresGrad: predictions.Requires,
		Name:         "mse_loss_result",
	}
	
	result, err := NewZerosTensor(1, 1, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	totalElements := predictions.Data.Rows * predictions.Data.Cols
	loss := 0.0
	for i := 0; i < predictions.Data.Rows; i++ {
		for j := 0; j < predictions.Data.Cols; j++ {
			diff := predictions.Data.Data[i][j] - targets.Data.Data[i][j]
			loss += diff * diff
		}
	}
	loss /= float64(totalElements)
	result.Data.Data[0][0] = loss
	
	// Set up backward function if gradient is required
	if predictions.Requires {
		result.Children = append(result.Children, predictions)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < predictions.Data.Rows; i++ {
				for j := 0; j < predictions.Data.Cols; j++ {
					diff := 2.0 * (predictions.Data.Data[i][j] - targets.Data.Data[i][j]) / float64(totalElements)
					predictions.Grad.Data[i][j] += diff * result.Grad.Data[0][0]
				}
			}
			return []*Tensor{predictions}
		}
	}
	
	return result, nil
}

// Transpose returns the transpose of a tensor with gradient tracking
func TensorTranspose(a *Tensor) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "transpose_result",
	}
	
	result, err := NewZerosTensor(a.Data.Cols, a.Data.Rows, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			result.Data.Data[j][i] = a.Data.Data[i][j]
		}
	}
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					a.Grad.Data[i][j] += result.Grad.Data[j][i]
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// Sum returns the sum of all elements in a tensor with gradient tracking
func Sum(a *Tensor) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "sum_result",
	}
	
	result, err := NewZerosTensor(1, 1, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	sum := 0.0
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			sum += a.Data.Data[i][j]
		}
	}
	result.Data.Data[0][0] = sum
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					a.Grad.Data[i][j] += result.Grad.Data[0][0]
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// Mean returns the mean of all elements in a tensor with gradient tracking
func Mean(a *Tensor) (*Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	
	config := &TensorConfig{
		RequiresGrad: a.Requires,
		Name:         "mean_result",
	}
	
	result, err := NewZerosTensor(1, 1, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create result tensor: %v", err)
	}
	
	// Forward pass
	totalElements := float64(a.Data.Rows * a.Data.Cols)
	sum := 0.0
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			sum += a.Data.Data[i][j]
		}
	}
	result.Data.Data[0][0] = sum / totalElements
	
	// Set up backward function if gradient is required
	if a.Requires {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() []*Tensor {
			for i := 0; i < a.Data.Rows; i++ {
				for j := 0; j < a.Data.Cols; j++ {
					a.Grad.Data[i][j] += result.Grad.Data[0][0] / totalElements
				}
			}
			return []*Tensor{a}
		}
	}
	
	return result, nil
}

// Clone creates a deep copy of a tensor
func (t *Tensor) Clone() (*Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("cannot clone nil tensor")
	}
	
	dataClone, err := t.Data.Clone()
	if err != nil {
		return nil, fmt.Errorf("failed to clone data matrix: %v", err)
	}
	
	var gradClone *Matrix
	if t.Grad != nil {
		gradClone, err = t.Grad.Clone()
		if err != nil {
			return nil, fmt.Errorf("failed to clone gradient matrix: %v", err)
		}
	}
	
	clone := &Tensor{
		Data:     dataClone,
		Grad:     gradClone,
		Requires: t.Requires,
		Name:     t.Name + "_clone",
	}
	
	return clone, nil
}

// Legacy compatibility functions to support older code
// These should be deprecated in future versions

// LegacyNewTensor creates a new tensor without error checking
func LegacyNewTensor(data *Matrix, requiresGrad bool) *Tensor {
	tensor, _ := NewTensor(data, &TensorConfig{RequiresGrad: requiresGrad})
	return tensor
}

// LegacyNewRandomTensor creates a new random tensor without error checking
func LegacyNewRandomTensor(rows, cols int, requiresGrad bool) *Tensor {
	tensor, _ := NewRandomTensor(rows, cols, &TensorConfig{RequiresGrad: requiresGrad})
	return tensor
}

// LegacyNewZerosTensor creates a new zeros tensor without error checking
func LegacyNewZerosTensor(rows, cols int, requiresGrad bool) *Tensor {
	tensor, _ := NewZerosTensor(rows, cols, &TensorConfig{RequiresGrad: requiresGrad})
	return tensor
}

// LegacyMatMul performs matrix multiplication without error checking
func LegacyMatMul(a, b *Tensor) *Tensor {
	result, _ := MatMul(a, b)
	return result
}

// LegacyAdd performs addition without error checking
func LegacyAdd(a, b *Tensor) *Tensor {
	result, _ := Add(a, b)
	return result
}

// LegacyReLU applies ReLU without error checking
func LegacyReLU(a *Tensor) *Tensor {
	result, _ := ReLU(a)
	return result
}

// LegacyGELU applies GELU without error checking
func LegacyGELU(a *Tensor) *Tensor {
	result, _ := GELU(a)
	return result
}

// LegacySoftmax applies softmax without error checking
func LegacySoftmax(a *Tensor) *Tensor {
	result, _ := Softmax(a)
	return result
}

// LegacyCrossEntropyLoss computes cross-entropy loss without error checking
func LegacyCrossEntropyLoss(logits *Tensor, targets []int) *Tensor {
	result, _ := CrossEntropyLoss(logits, targets)
	return result
}

// LegacyMSELoss computes MSE loss without error checking
func LegacyMSELoss(predictions, targets *Tensor) *Tensor {
	result, _ := MSELoss(predictions, targets)
	return result
}
