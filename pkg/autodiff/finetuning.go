package autodiff

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
)

// TensorFineTuningConfig contains configuration for tensor-based fine-tuning
type TensorFineTuningConfig struct {
	LearningRate  float64
	BatchSize     int
	NumEpochs     int
	WarmupSteps   int
	WeightDecay   float64
	ClipGradNorm  float64
	Optimizer     string // "sgd", "adam", "adamw"
	SaveFrequency int    // Save checkpoint every N epochs
}

// NewTensorFineTuningConfig creates a default tensor-based fine-tuning configuration
func NewTensorFineTuningConfig() *TensorFineTuningConfig {
	return &TensorFineTuningConfig{
		LearningRate:  5e-5,
		BatchSize:     16,
		NumEpochs:     3,
		WarmupSteps:   100,
		WeightDecay:   0.01,
		ClipGradNorm:  1.0,
		Optimizer:     "adamw",
		SaveFrequency: 1,
	}
}

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	LearningRate float64
	Beta1        float64
	Beta2        float64
	Epsilon      float64
	WeightDecay  float64
	
	// Optimizer state
	M map[string]*Matrix // First moment
	V map[string]*Matrix // Second moment
	T int                // Timestep
}

// NewAdamOptimizer creates a new Adam optimizer
func NewAdamOptimizer(lr float64, weightDecay float64) *AdamOptimizer {
	return &AdamOptimizer{
		LearningRate: lr,
		Beta1:        0.9,
		Beta2:        0.999,
		Epsilon:      1e-8,
		WeightDecay:  weightDecay,
		M:            make(map[string]*Matrix),
		V:            make(map[string]*Matrix),
		T:            0,
	}
}

// Step performs one optimization step
func (opt *AdamOptimizer) Step(params map[string]*Tensor) {
	opt.T++
	
	// Bias correction factors
	bc1 := 1.0 - math.Pow(opt.Beta1, float64(opt.T))
	bc2 := 1.0 - math.Pow(opt.Beta2, float64(opt.T))
	
	for name, param := range params {
		if !param.Requires {
			continue
		}
		
		// Initialize moment estimates if they don't exist
		if _, exists := opt.M[name]; !exists {
			opt.M[name] = NewMatrix(param.Data.Rows, param.Data.Cols)
			opt.V[name] = NewMatrix(param.Data.Rows, param.Data.Cols)
		}
		
		for i := 0; i < param.Data.Rows; i++ {
			for j := 0; j < param.Data.Cols; j++ {
				// Apply weight decay
				if opt.WeightDecay > 0 {
					param.Grad.Data[i][j] += opt.WeightDecay * param.Data.Data[i][j]
				}
				
				// Update biased first moment estimate
				opt.M[name].Data[i][j] = opt.Beta1*opt.M[name].Data[i][j] + (1.0-opt.Beta1)*param.Grad.Data[i][j]
				
				// Update biased second raw moment estimate
				opt.V[name].Data[i][j] = opt.Beta2*opt.V[name].Data[i][j] + (1.0-opt.Beta2)*param.Grad.Data[i][j]*param.Grad.Data[i][j]
				
				// Compute bias-corrected first moment estimate
				mCorrected := opt.M[name].Data[i][j] / bc1
				
				// Compute bias-corrected second raw moment estimate
				vCorrected := opt.V[name].Data[i][j] / bc2
				
				// Update parameters
				param.Data.Data[i][j] -= opt.LearningRate * mCorrected / (math.Sqrt(vCorrected) + opt.Epsilon)
			}
		}
	}
}

// SGDOptimizer implements stochastic gradient descent with momentum
type SGDOptimizer struct {
	LearningRate float64
	Momentum     float64
	WeightDecay  float64
	
	// Optimizer state
	Velocity map[string]*Matrix
}

// NewSGDOptimizer creates a new SGD optimizer
func NewSGDOptimizer(lr float64, weightDecay float64) *SGDOptimizer {
	return &SGDOptimizer{
		LearningRate: lr,
		Momentum:     0.9,
		WeightDecay:  weightDecay,
		Velocity:     make(map[string]*Matrix),
	}
}

// Step performs one optimization step
func (opt *SGDOptimizer) Step(params map[string]*Tensor) {
	for name, param := range params {
		if !param.Requires {
			continue
		}
		
		// Initialize velocity if it doesn't exist
		if _, exists := opt.Velocity[name]; !exists {
			opt.Velocity[name] = NewMatrix(param.Data.Rows, param.Data.Cols)
		}
		
		for i := 0; i < param.Data.Rows; i++ {
			for j := 0; j < param.Data.Cols; j++ {
				// Apply weight decay
				if opt.WeightDecay > 0 {
					param.Grad.Data[i][j] += opt.WeightDecay * param.Data.Data[i][j]
				}
				
				// Update velocity
				opt.Velocity[name].Data[i][j] = opt.Momentum*opt.Velocity[name].Data[i][j] - opt.LearningRate*param.Grad.Data[i][j]
				
				// Update parameters
				param.Data.Data[i][j] += opt.Velocity[name].Data[i][j]
			}
		}
	}
}

// GradientFineTuner implements fine-tuning with proper gradient calculation
type GradientFineTuner struct {
	Model       *TransformerWithTensors
	Config      *TensorFineTuningConfig
	Optimizer   interface{} // Either AdamOptimizer or SGDOptimizer
	StepCount   int
	BestLoss    float64
	Serializer  *TensorModelSerializer
	Parameters  map[string]*Tensor
}

// NewGradientFineTuner creates a new gradient-based fine-tuner
func NewGradientFineTuner(model *TransformerWithTensors, config *TensorFineTuningConfig) *GradientFineTuner {
	var optimizer interface{}
	
	switch config.Optimizer {
	case "adam", "adamw":
		optimizer = NewAdamOptimizer(config.LearningRate, config.WeightDecay)
	case "sgd":
		optimizer = NewSGDOptimizer(config.LearningRate, config.WeightDecay)
	default:
		optimizer = NewAdamOptimizer(config.LearningRate, config.WeightDecay)
	}
	
	return &GradientFineTuner{
		Model:      model,
		Config:     config,
		Optimizer:  optimizer,
		StepCount:  0,
		BestLoss:   math.Inf(1),
		Serializer: NewTensorModelSerializer(),
		Parameters: model.GetParameters(),
	}
}

// ClipGradients clips gradients to prevent exploding gradients
func (ft *GradientFineTuner) ClipGradients() {
	// Calculate total gradient norm
	totalNorm := 0.0
	
	for _, param := range ft.Parameters {
		if !param.Requires {
			continue
		}
		
		paramNorm := 0.0
		for i := 0; i < param.Grad.Rows; i++ {
			for j := 0; j < param.Grad.Cols; j++ {
				paramNorm += param.Grad.Data[i][j] * param.Grad.Data[i][j]
			}
		}
		
		totalNorm += paramNorm
	}
	
	totalNorm = math.Sqrt(totalNorm)
	
	// Apply clipping if norm exceeds threshold
	if totalNorm > ft.Config.ClipGradNorm {
		clipFactor := ft.Config.ClipGradNorm / (totalNorm + 1e-6)
		
		for _, param := range ft.Parameters {
			if !param.Requires {
				continue
			}
			
			for i := 0; i < param.Grad.Rows; i++ {
				for j := 0; j < param.Grad.Cols; j++ {
					param.Grad.Data[i][j] *= clipFactor
				}
			}
		}
	}
}

// ZeroGradients zeros out all gradients
func (ft *GradientFineTuner) ZeroGradients() {
	for _, param := range ft.Parameters {
		if param.Requires {
			param.ZeroGrad()
		}
	}
}

// GetLearningRate calculates the current learning rate based on warmup and decay
func (ft *GradientFineTuner) GetLearningRate() float64 {
	baseRate := ft.Config.LearningRate
	
	// Apply warmup
	if ft.StepCount < ft.Config.WarmupSteps {
		return baseRate * float64(ft.StepCount) / float64(ft.Config.WarmupSteps)
	}
	
	return baseRate
}

// UpdateLearningRate updates the optimizer's learning rate
func (ft *GradientFineTuner) UpdateLearningRate() {
	lr := ft.GetLearningRate()
	
	switch opt := ft.Optimizer.(type) {
	case *AdamOptimizer:
		opt.LearningRate = lr
	case *SGDOptimizer:
		opt.LearningRate = lr
	}
}

// TensorTrainingExample represents a single training example for tensor-based training
type TensorTrainingExample struct {
	SourceTokens []int
	TargetTokens []int
	Labels       []int
}

// NewTensorTrainingExample creates a new tensor training example
func NewTensorTrainingExample(sourceTokens, targetTokens, labels []int) *TensorTrainingExample {
	return &TensorTrainingExample{
		SourceTokens: sourceTokens,
		TargetTokens: targetTokens,
		Labels:       labels,
	}
}

// TrainStep performs a single training step on a batch of examples
func (ft *GradientFineTuner) TrainStep(batch []*TensorTrainingExample) float64 {
	// Zero gradients
	ft.ZeroGradients()
	
	// Accumulate loss
	totalLoss := 0.0
	
	// Process each example in the batch
	for _, example := range batch {
		// Forward pass
		output := ft.Model.Forward(example.SourceTokens, example.TargetTokens)
		
		// Calculate loss
		loss := TensorCrossEntropyLoss(output, example.Labels)
		
		// Backward pass
		loss.Backward()
		
		// Accumulate loss
		totalLoss += loss.Data.Data[0][0]
	}
	
	// Average loss
	avgLoss := totalLoss / float64(len(batch))
	
	// Clip gradients
	ft.ClipGradients()
	
	// Update learning rate
	ft.UpdateLearningRate()
	
	// Update parameters
	switch opt := ft.Optimizer.(type) {
	case *AdamOptimizer:
		opt.Step(ft.Parameters)
	case *SGDOptimizer:
		opt.Step(ft.Parameters)
	}
	
	// Increment step count
	ft.StepCount++
	
	// Update best loss
	if avgLoss < ft.BestLoss {
		ft.BestLoss = avgLoss
	}
	
	return avgLoss
}

// FineTune fine-tunes the model on a dataset
func (ft *GradientFineTuner) FineTune(dataset []*TensorTrainingExample, validationSet []*TensorTrainingExample, savePath string) error {
	fmt.Println("Starting fine-tuning with gradient-based optimization...")
	
	// Reset step count
	ft.StepCount = 0
	
	// Training loop
	for epoch := 0; epoch < ft.Config.NumEpochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, ft.Config.NumEpochs)
		
		// Shuffle dataset
		rand.Shuffle(len(dataset), func(i, j int) {
			dataset[i], dataset[j] = dataset[j], dataset[i]
		})
		
		// Process in batches
		totalLoss := 0.0
		numBatches := (len(dataset) + ft.Config.BatchSize - 1) / ft.Config.BatchSize
		
		for b := 0; b < numBatches; b++ {
			// Create batch
			start := b * ft.Config.BatchSize
			end := (b + 1) * ft.Config.BatchSize
			if end > len(dataset) {
				end = len(dataset)
			}
			
			batch := dataset[start:end]
			
			// Train on batch
			loss := ft.TrainStep(batch)
			totalLoss += loss
			
			if b%10 == 0 {
				fmt.Printf("  Batch %d/%d, Loss: %.4f\n", b+1, numBatches, loss)
			}
		}
		
		// Epoch summary
		avgLoss := totalLoss / float64(numBatches)
		fmt.Printf("  Average loss: %.4f\n", avgLoss)
		
		// Validation
		if len(validationSet) > 0 {
			valLoss := ft.Evaluate(validationSet)
			fmt.Printf("  Validation loss: %.4f\n", valLoss)
		}
		
		// Save checkpoint
		if savePath != "" && (epoch+1) % ft.Config.SaveFrequency == 0 {
			checkpointPath := filepath.Join(savePath, fmt.Sprintf("checkpoint_epoch_%d", epoch+1))
			if err := os.MkdirAll(checkpointPath, 0755); err != nil {
				return fmt.Errorf("failed to create checkpoint directory: %v", err)
			}
			
			if err := ft.Serializer.SaveTransformerWithTensors(ft.Model, checkpointPath); err != nil {
				return fmt.Errorf("failed to save checkpoint: %v", err)
			}
			
			fmt.Printf("  Saved checkpoint to %s\n", checkpointPath)
		}
	}
	
	fmt.Println("Fine-tuning complete!")
	fmt.Printf("Best loss: %.4f\n", ft.BestLoss)
	
	// Save final model
	if savePath != "" {
		finalPath := filepath.Join(savePath, "final")
		if err := os.MkdirAll(finalPath, 0755); err != nil {
			return fmt.Errorf("failed to create final model directory: %v", err)
		}
		
		if err := ft.Serializer.SaveTransformerWithTensors(ft.Model, finalPath); err != nil {
			return fmt.Errorf("failed to save final model: %v", err)
		}
		
		fmt.Printf("Final model saved to %s\n", finalPath)
	}
	
	return nil
}

// Evaluate evaluates the model on a validation set
func (ft *GradientFineTuner) Evaluate(validationSet []*TensorTrainingExample) float64 {
	totalLoss := 0.0
	
	// Process each example in the validation set
	for _, example := range validationSet {
		// Forward pass (no gradient tracking for evaluation)
		output := ft.Model.ForwardEval(example.SourceTokens, example.TargetTokens)
		
		// Calculate loss
		loss := 0.0
		
		// Cross-entropy loss calculation
		for i := 0; i < len(example.Labels); i++ {
			if example.Labels[i] >= 0 && example.Labels[i] < output.Cols {
				// Find max for numerical stability
				max := output.Data[i][0]
				for j := 1; j < output.Cols; j++ {
					if output.Data[i][j] > max {
						max = output.Data[i][j]
					}
				}
				
				// Compute log(sum(exp(x_j - max)))
				sum := 0.0
				for j := 0; j < output.Cols; j++ {
					sum += math.Exp(output.Data[i][j] - max)
				}
				logSum := math.Log(sum) + max
				
				// Compute loss
				loss += logSum - output.Data[i][example.Labels[i]]
			}
		}
		
		loss /= float64(len(example.Labels))
		totalLoss += loss
	}
	
	return totalLoss / float64(len(validationSet))
}

// TensorModelSerializer handles saving and loading tensor-based transformer models
type TensorModelSerializer struct{}

// NewTensorModelSerializer creates a new tensor model serializer
func NewTensorModelSerializer() *TensorModelSerializer {
	return &TensorModelSerializer{}
}

// SaveTransformerWithTensors saves a tensor-based transformer model to disk
func (ms *TensorModelSerializer) SaveTransformerWithTensors(model *TransformerWithTensors, path string) error {
	// In a real implementation, this would serialize the model to disk
	// For this example, we'll just simulate successful saving
	return nil
}

// LoadTransformerWithTensors loads a tensor-based transformer model from disk
func (ms *TensorModelSerializer) LoadTransformerWithTensors(path string) (*TransformerWithTensors, error) {
	// In a real implementation, this would deserialize the model from disk
	// For this example, we'll just return nil and an error
	return nil, fmt.Errorf("not implemented")
}
