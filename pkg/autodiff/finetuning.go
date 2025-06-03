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
	SegmentFilePaths []string // Paths to training data segment files
	// TrainingExamples can be used if SegmentFilePaths is empty/nil
	TrainingExamples []*TensorTrainingExample // Existing field for in-memory data
	MaxSeqLen        int                      // Max sequence length for tokenization
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
		SegmentFilePaths: nil,
		TrainingExamples: nil,
		MaxSeqLen:        512, // Default max sequence length
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
	"bufio"
	"io"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/transformer_reorganized/internal/tokenizer"
)

// TensorFineTuningConfig contains configuration for tensor-based fine-tuning
type TensorFineTuningConfig struct {
	LearningRate     float64
	BatchSize        int
	NumEpochs        int
	WarmupSteps      int
	WeightDecay      float64
	ClipGradNorm     float64
	Optimizer        string // "sgd", "adam", "adamw"
	SaveFrequency    int    // Save checkpoint every N epochs
	SegmentFilePaths []string // Paths to training data segment files
	TrainingExamples []*TensorTrainingExample // Existing field for in-memory data
	MaxSeqLen        int                      // Max sequence length for tokenization
	TokenizerVocabFile string                 // Path to tokenizer vocabulary (if needed for loading tokenizer)
}

// NewTensorFineTuningConfig creates a default tensor-based fine-tuning configuration
func NewTensorFineTuningConfig() *TensorFineTuningConfig {
	return &TensorFineTuningConfig{
		LearningRate:     5e-5,
		BatchSize:        16,
		NumEpochs:        3,
		WarmupSteps:      100,
		WeightDecay:      0.01,
		ClipGradNorm:     1.0,
		Optimizer:        "adamw",
		SaveFrequency:    1,
		SegmentFilePaths: nil,
		TrainingExamples: nil,
		MaxSeqLen:        512, // Default max sequence length
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
		if param.Grad == nil || !param.Requires { // Check Grad exists
			continue
		}

		// Initialize moment estimates if they don't exist
		if _, exists := opt.M[name]; !exists {
			opt.M[name], _ = NewMatrix(param.Data.Rows, param.Data.Cols) // Error handling omitted
			opt.V[name], _ = NewMatrix(param.Data.Rows, param.Data.Cols) // Error handling omitted
		}

		for i := 0; i < param.Data.Rows; i++ {
			for j := 0; j < param.Data.Cols; j++ {
				// Apply weight decay
				gradVal := param.Grad.Data[i][j]
				if opt.WeightDecay > 0 {
					gradVal += opt.WeightDecay * param.Data.Data[i][j]
				}

				// Update biased first moment estimate
				opt.M[name].Data[i][j] = opt.Beta1*opt.M[name].Data[i][j] + (1.0-opt.Beta1)*gradVal

				// Update biased second raw moment estimate
				opt.V[name].Data[i][j] = opt.Beta2*opt.V[name].Data[i][j] + (1.0-opt.Beta2)*gradVal*gradVal

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
		if param.Grad == nil || !param.Requires { // Check Grad exists
			continue
		}

		// Initialize velocity if it doesn't exist
		if _, exists := opt.Velocity[name]; !exists {
			opt.Velocity[name], _ = NewMatrix(param.Data.Rows, param.Data.Cols) // Error handling omitted
		}

		for i := 0; i < param.Data.Rows; i++ {
			for j := 0; j < param.Data.Cols; j++ {
				gradVal := param.Grad.Data[i][j]
				// Apply weight decay
				if opt.WeightDecay > 0 {
					gradVal += opt.WeightDecay * param.Data.Data[i][j]
				}

				// Update velocity
				opt.Velocity[name].Data[i][j] = opt.Momentum*opt.Velocity[name].Data[i][j] - opt.LearningRate*gradVal

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
	Tokenizer   *tokenizer.Tokenizer // Added Tokenizer
}

// NewGradientFineTuner creates a new gradient-based fine-tuner
func NewGradientFineTuner(model *TransformerWithTensors, config *TensorFineTuningConfig, tok *tokenizer.Tokenizer) *GradientFineTuner {
	var optimizer interface{}
	
	// Initialize tokenizer (simplified: assuming vocab is loaded into tokenizer already if needed)
	// A real implementation might load vocab here based on config.TokenizerVocabFile

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
		Tokenizer:  tok, // Store tokenizer
	}
}

// ClipGradients clips gradients to prevent exploding gradients
func (ft *GradientFineTuner) ClipGradients() {
	// Calculate total gradient norm
	totalNorm := 0.0
	
	for name, param := range ft.Parameters {
		if param.Grad == nil || !param.Requires { // Check Grad exists
			continue
		}
		
		paramNorm := 0.0
		for i := 0; i < param.Grad.Rows; i++ {
			for j := 0; j < param.Grad.Cols; j++ {
				paramNorm += param.Grad.Data[i][j] * param.Grad.Data[i][j]
			}
		}
		totalNorm += paramNorm // Sum of squares
	}
	totalNorm = math.Sqrt(totalNorm)

	if totalNorm > ft.Config.ClipGradNorm && ft.Config.ClipGradNorm > 0 { // Ensure ClipGradNorm is positive
		clipFactor := ft.Config.ClipGradNorm / (totalNorm + 1e-6) // Add epsilon to prevent division by zero
		for name, param := range ft.Parameters {
			if param.Grad == nil || !param.Requires { // Check Grad exists
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
		if param.Grad != nil && param.Requires { // Ensure Grad exists before trying to zero it
			err := param.ZeroGrad()
			if err != nil {
				// Log or handle error if ZeroGrad can fail, though current impl doesn't return error
				log.Printf("Warning: could not zero grad for param %s: %v (this might be a new error)", param.Name, err)
			}
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
// Modified to accept tokenized inputs directly if needed, or use examples from a struct
// For now, it directly uses TensorTrainingExample, assuming batching is done before calling.
func (ft *GradientFineTuner) TrainStep(batchExamples []*TensorTrainingExample) (float64, error) {
	ft.ZeroGradients()
	
	batchLoss := 0.0
	if len(batchExamples) == 0 {
		return 0.0, fmt.Errorf("cannot train on an empty batch")
	}

	for _, example := range batchExamples {
		// Forward pass
		// Assuming SourceTokens and TargetTokens are already prepared (e.g. for seq2seq)
		// For auto-encoding, TargetTokens might be same as SourceTokens or handled by loss
		logits, err := ft.Model.Forward(example.SourceTokens, example.TargetTokens)
		if err != nil {
			return 0, fmt.Errorf("forward pass failed: %v", err)
		}
		
		// Calculate loss
		// Labels should be prepared according to the task
		loss, err := CrossEntropyLoss(logits, example.Labels) // Using CrossEntropyLoss from autodiff.go
		if err != nil {
			return 0, fmt.Errorf("cross entropy loss calculation failed: %v", err)
		}
		
		// Backward pass
		err = loss.Backward()
		if err != nil {
			return 0, fmt.Errorf("backward pass failed: %v", err)
		}
		
		if loss.Data != nil && loss.Data.Rows > 0 && loss.Data.Cols > 0 {
			batchLoss += loss.Data.Data[0][0]
		} else {
			// This case should ideally not happen if loss calculation is correct
			log.Println("Warning: Loss tensor data is nil or empty.")
		}
	}
	
	avgLoss := batchLoss / float64(len(batchExamples))
	
	ft.ClipGradients()
	ft.UpdateLearningRate() // Update LR based on step count
	
	switch opt := ft.Optimizer.(type) {
	case *AdamOptimizer:
		opt.LearningRate = ft.GetLearningRate() // Ensure optimizer has current LR
		opt.Step(ft.Parameters)
	case *SGDOptimizer:
		opt.LearningRate = ft.GetLearningRate() // Ensure optimizer has current LR
		opt.Step(ft.Parameters)
	default:
		return 0, fmt.Errorf("unknown optimizer type")
	}
	
	ft.StepCount++
	return avgLoss, nil
}


// batchTensorTrainingExamples is a helper to create batches from a list of examples
func batchTensorTrainingExamples(examples []*TensorTrainingExample, batchSize int) [][]*TensorTrainingExample {
	var batches [][]*TensorTrainingExample
	for i := 0; i < len(examples); i += batchSize {
		end := i + batchSize
		if end > len(examples) {
			end = len(examples)
		}
		batches = append(batches, examples[i:end])
	}
	return batches
}


// loadAndTokenizeSegment reads a segment file, tokenizes lines, and creates training examples.
// Assumes auto-encoding: input is also the target/label.
func loadAndTokenizeSegment(filePath string, tokenizer *tokenizer.Tokenizer, maxSeqLen int) ([]*TensorTrainingExample, error) {
	if tokenizer == nil {
		return nil, fmt.Errorf("tokenizer is nil in loadAndTokenizeSegment")
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open segment file %s: %v", filePath, err)
	}
	defer file.Close()

	var examples []*TensorTrainingExample
	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		text := scanner.Text()
		if text == "" {
			continue
		}

		// Encode text. Using default encode options for simplicity here.
		// A more robust implementation might pass encodeOptions or configure them.
		// For auto-encoding, source, target, and labels are derived from the same input text.
		// MaxLength for tokenizer Encode is set by maxSeqLen
		encodeOptions := tokenizer.DefaultEncodeOptions()
		if maxSeqLen > 0 {
			encodeOptions.MaxLength = maxSeqLen
			encodeOptions.Truncation = true // Enable truncation if maxSeqLen is set
		}


		encodingResult, err := tokenizer.Encode(text, encodeOptions)
		if err != nil {
			log.Printf("Warning: Failed to tokenize line %d in %s: %v. Skipping.", lineNum, filePath, err)
			continue
		}

		// For auto-encoding: input_ids are source, target, and also serve as labels for loss.
		// The CrossEntropyLoss expects labels to be indices, so this needs careful handling.
		// If model output is (batch, seq_len, vocab_size), labels should be (batch, seq_len).
		// Here, input_ids are (seq_len). We assume the loss function handles this.
		// This is a simplified setup for auto-encoding a sequence.
		// For a typical transformer language model, target/labels would be input_ids shifted by one.
		// For this example, we'll keep it simple: SourceTokens = input_ids, TargetTokens = input_ids (for decoder input), Labels = input_ids (for loss calculation)
		// This implies the decoder's task is to reconstruct the input.

		// Adjusting for typical seq2seq autoencoding:
		// SourceTokens: BOS + sequence
		// TargetTokens: sequence (for teacher forcing during training, decoder input)
		// Labels: sequence + EOS (what the decoder should predict)
		// For simplicity now, we will use encodingResult.InputIDs for all.
		// A more correct auto-encoding setup would be:
		// srcTokens = encodingResult.InputIDs
		// tgtTokens = encodingResult.InputIDs (usually starts with BOS, ends before final EOS for input)
		// labels = encodingResult.InputIDs (usually starts after BOS, includes final EOS for output)
		// For now, we just use InputIDs as is.

		// Simple auto-encoding: predict the same sequence.
		// Ensure that labels are not longer than MaxSeqLen if model outputs fixed length.
		// If model.Forward produces logits of shape (batch, seq_len, vocab_size),
		// then labels should be of shape (batch, seq_len).
		// encodingResult.InputIDs already respects MaxSeqLen due to tokenizer.Encode.
		example := &TensorTrainingExample{
			SourceTokens: encodingResult.InputIDs,
			TargetTokens: encodingResult.InputIDs, // For auto-encoder, decoder input can be same as source.
			Labels:       encodingResult.InputIDs, // Labels are the tokens to predict.
		}
		examples = append(examples, example)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading segment file %s: %v", filePath, err)
	}

	return examples, nil
}


// FineTune fine-tunes the model on a dataset
func (ft *GradientFineTuner) FineTune(validationSet []*TensorTrainingExample, savePath string) error { // Removed dataset param
	fmt.Println("Starting fine-tuning with gradient-based optimization...")
	if ft.Tokenizer == nil && len(ft.Config.SegmentFilePaths) > 0 {
		return fmt.Errorf("tokenizer is required for segmented file loading but is nil")
	}
	if ft.Config.MaxSeqLen <= 0 && len(ft.Config.SegmentFilePaths) > 0 {
		// Or load from tokenizer if it has a max length property
		return fmt.Errorf("MaxSeqLen must be positive when using segmented file loading")
	}


	rand.Seed(time.Now().UnixNano()) // Seed random number generator for shuffling

	// Training loop
	for epoch := 0; epoch < ft.Config.NumEpochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, ft.Config.NumEpochs)
		epochLoss := 0.0
		numBatchesProcessed := 0

		currentTrainingExamples := []*TensorTrainingExample{}

		if len(ft.Config.SegmentFilePaths) > 0 {
			// Segmented file loading
			segmentPaths := make([]string, len(ft.Config.SegmentFilePaths))
			copy(segmentPaths, ft.Config.SegmentFilePaths)
			rand.Shuffle(len(segmentPaths), func(i, j int) { segmentPaths[i], segmentPaths[j] = segmentPaths[j], segmentPaths[i] })

			for _, filePath := range segmentPaths {
				log.Printf("Processing segment: %s for epoch %d\n", filePath, epoch+1)
				segmentExamples, err := loadAndTokenizeSegment(filePath, ft.Tokenizer, ft.Config.MaxSeqLen)
				if err != nil {
					log.Printf("Warning: Failed to load or tokenize segment %s: %v. Skipping.", filePath, err)
					continue
				}
				if len(segmentExamples) == 0 {
					log.Printf("Warning: No examples found in segment %s. Skipping.", filePath)
					continue
				}

				// Intra-segment shuffling
				rand.Shuffle(len(segmentExamples), func(i, j int) { segmentExamples[i], segmentExamples[j] = segmentExamples[j], segmentExamples[i] })

				// Process batches from this segment
				segmentBatches := batchTensorTrainingExamples(segmentExamples, ft.Config.BatchSize)
				for i, batch := range segmentBatches {
					loss, trainErr := ft.TrainStep(batch)
					if trainErr != nil {
						log.Printf("Error during training step for batch %d from segment %s: %v. Skipping batch.", i, filePath, trainErr)
						continue
					}
					epochLoss += loss
					numBatchesProcessed++
					if numBatchesProcessed%10 == 0 { // Log progress periodically
						fmt.Printf("  Epoch %d, Segment %s, Batch %d/%d (approx), Current Batch Loss: %.4f, Avg Epoch Loss: %.4f\n",
							epoch+1, filepath.Base(filePath), i+1, len(segmentBatches), loss, epochLoss/float64(numBatchesProcessed))
					}
				}
			}
		} else if len(ft.Config.TrainingExamples) > 0 {
			// In-memory data loading
			currentTrainingExamples = make([]*TensorTrainingExample, len(ft.Config.TrainingExamples))
			copy(currentTrainingExamples, ft.Config.TrainingExamples)
			rand.Shuffle(len(currentTrainingExamples), func(i, j int) { currentTrainingExamples[i], currentTrainingExamples[j] = currentTrainingExamples[j], currentTrainingExamples[i] })
			
			inMemoryBatches := batchTensorTrainingExamples(currentTrainingExamples, ft.Config.BatchSize)
			for i, batch := range inMemoryBatches {
				loss, trainErr := ft.TrainStep(batch)
				if trainErr != nil {
						log.Printf("Error during training step for in-memory batch %d: %v. Skipping batch.", i, trainErr)
						continue
				}
				epochLoss += loss
				numBatchesProcessed++
				if numBatchesProcessed%10 == 0 {
					fmt.Printf("  Epoch %d, In-memory Batch %d/%d, Current Batch Loss: %.4f, Avg Epoch Loss: %.4f\n",
						epoch+1, i+1, len(inMemoryBatches), loss, epochLoss/float64(numBatchesProcessed))
				}
			}
		} else {
			fmt.Println("No training data provided (neither SegmentFilePaths nor TrainingExamples). Skipping epoch.")
			continue
		}

		// Epoch summary
		if numBatchesProcessed > 0 {
			avgEpochLoss := epochLoss / float64(numBatchesProcessed)
			fmt.Printf("Epoch %d Summary: Average Loss: %.4f, Total Batches: %d\n", epoch+1, avgEpochLoss, numBatchesProcessed)
			if avgEpochLoss < ft.BestLoss {
				ft.BestLoss = avgEpochLoss
				// Potentially save model if it's the best so far, based on epoch loss
			}
		} else {
			fmt.Printf("Epoch %d Summary: No batches were processed.\n", epoch+1)
		}
		
		// Validation (if validationSet is provided)
		if len(validationSet) > 0 {
			valLoss, valErr := ft.Evaluate(validationSet)
			if valErr != nil {
				log.Printf("Error during validation for epoch %d: %v", epoch+1, valErr)
			} else {
				fmt.Printf("  Epoch %d Validation Loss: %.4f\n", epoch+1, valLoss)
			}
		}
		
		// Save checkpoint
		if savePath != "" && (epoch+1)%ft.Config.SaveFrequency == 0 {
			checkpointPath := filepath.Join(savePath, fmt.Sprintf("checkpoint_epoch_%d", epoch+1))
			if err := os.MkdirAll(checkpointPath, 0755); err != nil {
				log.Printf("Failed to create checkpoint directory %s: %v", checkpointPath, err)
			} else {
				if err := ft.Serializer.SaveTransformerWithTensors(ft.Model, checkpointPath); err != nil { // Assuming serializer exists and is configured
					log.Printf("Failed to save checkpoint to %s: %v", checkpointPath, err)
				} else {
					fmt.Printf("  Saved checkpoint to %s\n", checkpointPath)
				}
			}
		}
	}
	
	fmt.Println("Fine-tuning complete!")
	fmt.Printf("Best training loss achieved: %.4f\n", ft.BestLoss)
	
	// Save final model
	if savePath != "" {
		finalModelPath := filepath.Join(savePath, "final_model")
		if err := os.MkdirAll(finalModelPath, 0755); err != nil {
			log.Printf("Failed to create final model directory %s: %v", finalModelPath, err)
		} else {
			if err := ft.Serializer.SaveTransformerWithTensors(ft.Model, finalModelPath); err != nil {
				log.Printf("Failed to save final model to %s: %v", finalModelPath, err)
			} else {
				fmt.Printf("Final model saved to %s\n", finalModelPath)
			}
		}
	}
	
	return nil
}


// Evaluate evaluates the model on a validation set
func (ft *GradientFineTuner) Evaluate(validationSet []*TensorTrainingExample) (float64, error) {
	if len(validationSet) == 0 {
		return 0.0, nil // No validation data to evaluate
	}
	totalLoss := 0.0
	numExamples := 0

	validationBatches := batchTensorTrainingExamples(validationSet, ft.Config.BatchSize)

	for _, batch := range validationBatches {
		if len(batch) == 0 { continue }

		for _, example := range batch {
			// Forward pass (no gradient tracking for evaluation via model.ForwardEval)
			// Model.ForwardEval should return Matrix for direct calculation or handle tensor internally.
			// For simplicity, assume ForwardEval returns logits as Matrix.
			// If it returns Tensor, then loss calculation needs to be Tensor-based or convert.
			logitsMatrix, err := ft.Model.ForwardEval(example.SourceTokens, example.TargetTokens)
			if err != nil {
				log.Printf("Error during ForwardEval for validation: %v. Skipping example.", err)
				continue
			}

			// Calculate loss using the matrix output.
			// This is a simplified cross-entropy for matrix logits.
			// A proper implementation might need a matrix version of CrossEntropyLoss or convert logitsMatrix to Tensor.
			exampleLoss := 0.0
			numValidLabels := 0
			for i := 0; i < len(example.Labels); i++ { // Assuming labels correspond to rows in logitsMatrix
				if example.Labels[i] >= 0 && example.Labels[i] < logitsMatrix.Cols {
					max := logitsMatrix.Data[i][0]
					for j := 1; j < logitsMatrix.Cols; j++ {
						if logitsMatrix.Data[i][j] > max {
							max = logitsMatrix.Data[i][j]
						}
					}
					sumExp := 0.0
					for j := 0; j < logitsMatrix.Cols; j++ {
						sumExp += math.Exp(logitsMatrix.Data[i][j] - max)
					}
					logSumExp := math.Log(sumExp) + max
					exampleLoss += (logSumExp - logitsMatrix.Data[i][example.Labels[i]])
					numValidLabels++
				}
			}
			if numValidLabels > 0 {
				totalLoss += exampleLoss / float64(numValidLabels)
				numExamples++
			}
		}
	}
	
	if numExamples == 0 {
		return 0.0, fmt.Errorf("no valid examples processed during evaluation")
	}
	return totalLoss / float64(numExamples), nil
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
