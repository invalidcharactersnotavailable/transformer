package transformer

import (
	"math"
)

// Tensor represents a tensor with gradient tracking capabilities
type Tensor struct {
	Data      *Matrix
	Grad      *Matrix
	Requires  bool
	BackwardFn func() []*Tensor
	Children  []*Tensor
}

// NewTensor creates a new tensor from a matrix
func NewTensor(data *Matrix, requiresGrad bool) *Tensor {
	grad := NewMatrix(data.Rows, data.Cols)
	return &Tensor{
		Data:      data,
		Grad:      grad,
		Requires:  requiresGrad,
		BackwardFn: nil,
		Children:  make([]*Tensor, 0),
	}
}

// NewRandomTensor creates a new tensor with random values
func NewRandomTensor(rows, cols int, requiresGrad bool) *Tensor {
	return NewTensor(NewRandomMatrix(rows, cols), requiresGrad)
}

// NewZerosTensor creates a new tensor filled with zeros
func NewZerosTensor(rows, cols int, requiresGrad bool) *Tensor {
	return NewTensor(NewMatrix(rows, cols), requiresGrad)
}

// ZeroGrad zeros out the gradient
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := 0; i < t.Grad.Rows; i++ {
			for j := 0; j < t.Grad.Cols; j++ {
				t.Grad.Data[i][j] = 0.0
			}
		}
	}
}

// Backward computes gradients
func (t *Tensor) Backward() {
	// Initialize gradient to 1.0 if it's a scalar
	if t.Data.Rows == 1 && t.Data.Cols == 1 {
		t.Grad.Data[0][0] = 1.0
	}
	
	// Topological sort for backward pass
	visited := make(map[*Tensor]bool)
	topo := make([]*Tensor, 0)
	
	var buildTopo func(node *Tensor)
	buildTopo = func(node *Tensor) {
		if visited[node] {
			return
		}
		visited[node] = true
		for _, child := range node.Children {
			buildTopo(child)
		}
		topo = append(topo, node)
	}
	
	buildTopo(t)
	
	// Backward pass
	for i := len(topo) - 1; i >= 0; i-- {
		node := topo[i]
		if node.BackwardFn != nil {
			parents := node.BackwardFn()
			for _, parent := range parents {
				if parent.Requires {
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
}

// MatMul performs matrix multiplication with gradient tracking
func TensorMatMul(a, b *Tensor) *Tensor {
	if a.Data.Cols != b.Data.Rows {
		panic("Matrix dimensions don't match for multiplication")
	}
	
	result := NewZerosTensor(a.Data.Rows, b.Data.Cols, a.Requires || b.Requires)
	
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
				bT := Transpose(b.Data)
				dA := MatMul(result.Grad, bT)
				for i := 0; i < dA.Rows; i++ {
					for j := 0; j < dA.Cols; j++ {
						a.Grad.Data[i][j] += dA.Data[i][j]
					}
				}
			}
			
			if b.Requires {
				// dL/dB = A^T * dL/dC
				aT := Transpose(a.Data)
				dB := MatMul(aT, result.Grad)
				for i := 0; i < dB.Rows; i++ {
					for j := 0; j < dB.Cols; j++ {
						b.Grad.Data[i][j] += dB.Data[i][j]
					}
				}
			}
			
			return []*Tensor{a, b}
		}
	}
	
	return result
}

// Add performs element-wise addition with gradient tracking
func TensorAdd(a, b *Tensor) *Tensor {
	if a.Data.Rows != b.Data.Rows || a.Data.Cols != b.Data.Cols {
		panic("Matrix dimensions don't match for addition")
	}
	
	result := NewZerosTensor(a.Data.Rows, a.Data.Cols, a.Requires || b.Requires)
	
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
	
	return result
}

// ReLU applies the ReLU activation function with gradient tracking
func TensorReLU(a *Tensor) *Tensor {
	result := NewZerosTensor(a.Data.Rows, a.Data.Cols, a.Requires)
	
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
	
	return result
}

// GELU applies the GELU activation function with gradient tracking
func TensorGELU(a *Tensor) *Tensor {
	result := NewZerosTensor(a.Data.Rows, a.Data.Cols, a.Requires)
	
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
	
	return result
}

// Softmax applies the softmax function with gradient tracking
func TensorSoftmax(a *Tensor) *Tensor {
	result := NewZerosTensor(a.Data.Rows, a.Data.Cols, a.Requires)
	
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
	
	return result
}

// CrossEntropyLoss computes the cross-entropy loss with gradient tracking
func TensorCrossEntropyLoss(logits *Tensor, targets []int) *Tensor {
	batchSize := logits.Data.Rows
	result := NewZerosTensor(1, 1, logits.Requires)
	
	// Forward pass
	loss := 0.0
	for i := 0; i < batchSize; i++ {
		if targets[i] >= 0 && targets[i] < logits.Data.Cols {
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
	}
	loss /= float64(batchSize)
	result.Data.Data[0][0] = loss
	
	// Set up backward function if gradient is required
	if logits.Requires {
		result.Children = append(result.Children, logits)
		result.BackwardFn = func() []*Tensor {
			// Compute softmax gradients
			for i := 0; i < batchSize; i++ {
				if targets[i] >= 0 && targets[i] < logits.Data.Cols {
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
			}
			return []*Tensor{logits}
		}
	}
	
	return result
}

// MSELoss computes the mean squared error loss with gradient tracking
func TensorMSELoss(predictions *Tensor, targets *Tensor) *Tensor {
	if predictions.Data.Rows != targets.Data.Rows || predictions.Data.Cols != targets.Data.Cols {
		panic("Predictions and targets dimensions don't match")
	}
	
	result := NewZerosTensor(1, 1, predictions.Requires)
	
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
	
	return result
}
