package transformer

import (
	"fmt"
	"math"
	"testing"
)

// TestAutodiffGradients tests basic gradient calculation
func TestAutodiffGradients(t *testing.T) {
	// Create tensors
	a := NewRandomTensor(2, 3, true)
	b := NewRandomTensor(3, 2, true)
	
	// Forward pass
	c := TensorMatMul(a, b)
	
	// Initialize gradient
	c.Grad.Data[0][0] = 1.0
	c.Grad.Data[0][1] = 1.0
	c.Grad.Data[1][0] = 1.0
	c.Grad.Data[1][1] = 1.0
	
	// Backward pass
	c.Backward()
	
	// Check that gradients are non-zero
	hasGradient := false
	for i := 0; i < a.Grad.Rows; i++ {
		for j := 0; j < a.Grad.Cols; j++ {
			if a.Grad.Data[i][j] != 0 {
				hasGradient = true
				break
			}
		}
	}
	
	if !hasGradient {
		t.Errorf("Expected non-zero gradients in tensor a")
	}
	
	hasGradient = false
	for i := 0; i < b.Grad.Rows; i++ {
		for j := 0; j < b.Grad.Cols; j++ {
			if b.Grad.Data[i][j] != 0 {
				hasGradient = true
				break
			}
		}
	}
	
	if !hasGradient {
		t.Errorf("Expected non-zero gradients in tensor b")
	}
}

// TestCrossEntropyLoss tests cross-entropy loss and its gradients
func TestCrossEntropyLoss(t *testing.T) {
	// Create logits tensor
	logits := NewZerosTensor(2, 3, true)
	
	// Set some values
	logits.Data.Data[0][0] = 0.1
	logits.Data.Data[0][1] = 0.2
	logits.Data.Data[0][2] = 0.3
	logits.Data.Data[1][0] = 0.5
	logits.Data.Data[1][1] = 0.1
	logits.Data.Data[1][2] = 0.2
	
	// Create targets
	targets := []int{1, 2}
	
	// Calculate loss
	loss := TensorCrossEntropyLoss(logits, targets)
	
	// Check that loss is reasonable
	if loss.Data.Data[0][0] <= 0 {
		t.Errorf("Expected positive loss, got %f", loss.Data.Data[0][0])
	}
	
	// Backward pass
	loss.Backward()
	
	// Check that gradients are non-zero
	hasGradient := false
	for i := 0; i < logits.Grad.Rows; i++ {
		for j := 0; j < logits.Grad.Cols; j++ {
			if logits.Grad.Data[i][j] != 0 {
				hasGradient = true
				break
			}
		}
	}
	
	if !hasGradient {
		t.Errorf("Expected non-zero gradients in logits")
	}
}

// TestAdamOptimizer tests the Adam optimizer
func TestAdamOptimizer(t *testing.T) {
	// Create parameters
	params := make(map[string]*Tensor)
	params["w1"] = NewRandomTensor(2, 3, true)
	params["w2"] = NewRandomTensor(3, 2, true)
	
	// Set gradients
	for i := 0; i < params["w1"].Grad.Rows; i++ {
		for j := 0; j < params["w1"].Grad.Cols; j++ {
			params["w1"].Grad.Data[i][j] = 0.1
		}
	}
	
	for i := 0; i < params["w2"].Grad.Rows; i++ {
		for j := 0; j < params["w2"].Grad.Cols; j++ {
			params["w2"].Grad.Data[i][j] = 0.2
		}
	}
	
	// Create optimizer
	optimizer := NewAdamOptimizer(0.001, 0.01)
	
	// Store original parameters
	w1Original := make([][]float64, params["w1"].Data.Rows)
	for i := 0; i < params["w1"].Data.Rows; i++ {
		w1Original[i] = make([]float64, params["w1"].Data.Cols)
		for j := 0; j < params["w1"].Data.Cols; j++ {
			w1Original[i][j] = params["w1"].Data.Data[i][j]
		}
	}
	
	// Step
	optimizer.Step(params)
	
	// Check that parameters have been updated
	changed := false
	for i := 0; i < params["w1"].Data.Rows; i++ {
		for j := 0; j < params["w1"].Data.Cols; j++ {
			if params["w1"].Data.Data[i][j] != w1Original[i][j] {
				changed = true
				break
			}
		}
	}
	
	if !changed {
		t.Errorf("Expected parameters to change after optimizer step")
	}
}

// TestSimpleTraining tests a simple training loop
func TestSimpleTraining(t *testing.T) {
	// Create a small transformer model
	model := NewTransformerWithTensors(10, 8, 1, 2, 16, 50)
	
	// Create a fine-tuning config
	config := NewTensorFineTuningConfig()
	config.NumEpochs = 2
	config.BatchSize = 2
	
	// Create a fine-tuner
	fineTuner := NewGradientFineTuner(model, config)
	
	// Create a simple dataset
	dataset := []*TensorTrainingExample{
		NewTensorTrainingExample([]int{1, 2, 3}, []int{4, 5}, []int{5, 6}),
		NewTensorTrainingExample([]int{2, 3, 4}, []int{5, 6}, []int{6, 7}),
		NewTensorTrainingExample([]int{3, 4, 5}, []int{6, 7}, []int{7, 8}),
		NewTensorTrainingExample([]int{4, 5, 6}, []int{7, 8}, []int{8, 9}),
	}
	
	// Train for one step
	batch := dataset[:2]
	initialLoss := fineTuner.TrainStep(batch)
	
	// Train for another step
	fineTuner.ZeroGradients()
	finalLoss := fineTuner.TrainStep(batch)
	
	// Check that loss decreased
	fmt.Printf("Initial loss: %f, Final loss: %f\n", initialLoss, finalLoss)
	if finalLoss >= initialLoss {
		t.Errorf("Expected loss to decrease, got initial: %f, final: %f", initialLoss, finalLoss)
	}
}

// TestGradientClipping tests gradient clipping
func TestGradientClipping(t *testing.T) {
	// Create a small transformer model
	model := NewTransformerWithTensors(10, 8, 1, 2, 16, 50)
	
	// Create a fine-tuning config with small clip norm
	config := NewTensorFineTuningConfig()
	config.ClipGradNorm = 0.1
	
	// Create a fine-tuner
	fineTuner := NewGradientFineTuner(model, config)
	
	// Set large gradients
	for _, param := range fineTuner.Parameters {
		if param.Requires {
			for i := 0; i < param.Grad.Rows; i++ {
				for j := 0; j < param.Grad.Cols; j++ {
					param.Grad.Data[i][j] = 1.0
				}
			}
		}
	}
	
	// Calculate initial gradient norm
	initialNorm := 0.0
	for _, param := range fineTuner.Parameters {
		if param.Requires {
			for i := 0; i < param.Grad.Rows; i++ {
				for j := 0; j < param.Grad.Cols; j++ {
					initialNorm += param.Grad.Data[i][j] * param.Grad.Data[i][j]
				}
			}
		}
	}
	initialNorm = math.Sqrt(initialNorm)
	
	// Clip gradients
	fineTuner.ClipGradients()
	
	// Calculate final gradient norm
	finalNorm := 0.0
	for _, param := range fineTuner.Parameters {
		if param.Requires {
			for i := 0; i < param.Grad.Rows; i++ {
				for j := 0; j < param.Grad.Cols; j++ {
					finalNorm += param.Grad.Data[i][j] * param.Grad.Data[i][j]
				}
			}
		}
	}
	finalNorm = math.Sqrt(finalNorm)
	
	// Check that norm was clipped
	fmt.Printf("Initial norm: %f, Final norm: %f, Clip threshold: %f\n", initialNorm, finalNorm, config.ClipGradNorm)
	if finalNorm > config.ClipGradNorm*1.01 { // Allow small margin for floating point error
		t.Errorf("Expected gradient norm to be clipped to %f, got %f", config.ClipGradNorm, finalNorm)
	}
}

// TestLearningRateSchedule tests the learning rate schedule
func TestLearningRateSchedule(t *testing.T) {
	// Create a small transformer model
	model := NewTransformerWithTensors(10, 8, 1, 2, 16, 50)
	
	// Create a fine-tuning config with warmup
	config := NewTensorFineTuningConfig()
	config.LearningRate = 0.001
	config.WarmupSteps = 10
	
	// Create a fine-tuner
	fineTuner := NewGradientFineTuner(model, config)
	
	// Check initial learning rate (should be 0 at step 0)
	initialLR := fineTuner.GetLearningRate()
	if initialLR != 0 {
		t.Errorf("Expected initial learning rate to be 0, got %f", initialLR)
	}
	
	// Set step count to 5 (middle of warmup)
	fineTuner.StepCount = 5
	midWarmupLR := fineTuner.GetLearningRate()
	if midWarmupLR <= 0 || midWarmupLR >= config.LearningRate {
		t.Errorf("Expected mid-warmup learning rate to be between 0 and %f, got %f", config.LearningRate, midWarmupLR)
	}
	
	// Set step count to warmup steps (end of warmup)
	fineTuner.StepCount = config.WarmupSteps
	fullLR := fineTuner.GetLearningRate()
	if math.Abs(fullLR-config.LearningRate) > 1e-6 {
		t.Errorf("Expected full learning rate to be %f, got %f", config.LearningRate, fullLR)
	}
}
