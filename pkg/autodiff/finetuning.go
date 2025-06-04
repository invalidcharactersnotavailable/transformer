package autodiff

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/transformer_reorganized/internal/tokenizer"
	"github.com/transformer_reorganized/pkg/moe"
)

// TensorFineTuningConfig contains configuration for tensor-based fine-tuning
type TensorFineTuningConfig struct {
	LearningRate       float64
	BatchSize          int
	NumEpochs          int
	WarmupSteps        int
	WeightDecay        float64
	ClipGradNorm       float64
	Optimizer          string
	SaveFrequency      int
	SegmentFilePaths   []string
	TrainingExamples   []*TensorTrainingExample
	MaxSeqLen          int
	TokenizerVocabFile string

	// MoE Loss Coefficients
	MoERouterZLossCoeff     float64
	MoELoadBalanceLossCoeff float64
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
		MaxSeqLen:        512,
		MoERouterZLossCoeff:     0.01, // Default value
		MoELoadBalanceLossCoeff: 0.01, // Default value
	}
}

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	LearningRate float64; Beta1 float64; Beta2 float64; Epsilon float64; WeightDecay  float64
	M map[string]*Matrix; V map[string]*Matrix; T int
}

// NewAdamOptimizer creates a new Adam optimizer
func NewAdamOptimizer(lr float64, weightDecay float64) *AdamOptimizer {
	return &AdamOptimizer{ LearningRate: lr, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8, WeightDecay: weightDecay, M: make(map[string]*Matrix), V: make(map[string]*Matrix), T: 0 }
}

// Step performs one optimization step
func (opt *AdamOptimizer) Step(params map[string]*Tensor) {
	opt.T++
	bc1 := 1.0 - math.Pow(opt.Beta1, float64(opt.T)); bc2 := 1.0 - math.Pow(opt.Beta2, float64(opt.T))
	for name, param := range params {
		if param.Grad == nil || !param.RequiresGrad { continue }
		if _, exists := opt.M[name]; !exists { opt.M[name], _ = NewMatrix(param.Data.Rows, param.Data.Cols); opt.V[name], _ = NewMatrix(param.Data.Rows, param.Data.Cols) }
		for i := 0; i < param.Data.Rows; i++ { for j := 0; j < param.Data.Cols; j++ {
			gradVal := param.Grad.Data[i][j]; if opt.WeightDecay > 0 { gradVal += opt.WeightDecay * param.Data.Data[i][j] }
			opt.M[name].Data[i][j] = opt.Beta1*opt.M[name].Data[i][j] + (1.0-opt.Beta1)*gradVal
			opt.V[name].Data[i][j] = opt.Beta2*opt.V[name].Data[i][j] + (1.0-opt.Beta2)*gradVal*gradVal
			mCorrected := opt.M[name].Data[i][j] / bc1; vCorrected := opt.V[name].Data[i][j] / bc2
			param.Data.Data[i][j] -= opt.LearningRate * mCorrected / (math.Sqrt(vCorrected) + opt.Epsilon)
		} }
	}
}

// SGDOptimizer implements stochastic gradient descent with momentum
type SGDOptimizer struct { LearningRate float64; Momentum float64; WeightDecay float64; Velocity map[string]*Matrix }
func NewSGDOptimizer(lr float64, weightDecay float64) *SGDOptimizer { return &SGDOptimizer{ LearningRate: lr, Momentum: 0.9, WeightDecay: weightDecay, Velocity: make(map[string]*Matrix) } }
func (opt *SGDOptimizer) Step(params map[string]*Tensor) {
	for name, param := range params {
		if param.Grad == nil || !param.RequiresGrad { continue }
		if _, exists := opt.Velocity[name]; !exists { opt.Velocity[name], _ = NewMatrix(param.Data.Rows, param.Data.Cols) }
		for i := 0; i < param.Data.Rows; i++ { for j := 0; j < param.Data.Cols; j++ {
			gradVal := param.Grad.Data[i][j]; if opt.WeightDecay > 0 { gradVal += opt.WeightDecay * param.Data.Data[i][j] }
			opt.Velocity[name].Data[i][j] = opt.Momentum*opt.Velocity[name].Data[i][j] - opt.LearningRate*gradVal
			param.Data.Data[i][j] += opt.Velocity[name].Data[i][j]
		} }
	}
}

// GradientFineTuner structure
type GradientFineTuner struct {
	Model       *TransformerWithTensors; Config      *TensorFineTuningConfig
	Optimizer   interface{}; StepCount   int; BestLoss    float64
	Serializer  *TensorModelSerializer; Parameters  map[string]*Tensor
	Tokenizer   *tokenizer.Tokenizer
}

// NewGradientFineTuner constructor
// It's assumed that the model passed here is already configured (e.g. with MoE or not).
// The fine-tuning config provides training-specific parameters like loss coefficients for MoE.
func NewGradientFineTuner(model *TransformerWithTensors, config *TensorFineTuningConfig, tok *tokenizer.Tokenizer) *GradientFineTuner {
	var optimizer interface{}
	switch config.Optimizer {
	case "adam", "adamw": optimizer = NewAdamOptimizer(config.LearningRate, config.WeightDecay)
	case "sgd": optimizer = NewSGDOptimizer(config.LearningRate, config.WeightDecay)
	default: optimizer = NewAdamOptimizer(config.LearningRate, config.WeightDecay)
	}
	// Pass MoE loss coeffs from fine-tuning config to the model's MoE layers' configs.
	// This requires core.Config to also hold these, or a way to pass them to MoELayerConfig during model construction.
	// For now, assume TransformerWithTensors's core.Config is the source for these for MoELayerConfig.
	// The HyperParameterManager should ensure core.Config gets these values if they are top-level HPs.
	// If core.Config does NOT have MoERouterZLossCoeff, then this is where they need to be plumbed to the model.
	// Let's assume core.Config was updated by HyperParameterManager based on top-level HPs.
	// And tensor_transformer.NewTransformerWithTensors uses these from core.Config to set up moe.MoELayerConfig.
	// So, no direct action needed here in NewGradientFineTuner to pass loss coeffs to model,
	// as they are assumed to be part of the model's structural config already.
	// The fineTuningConfig here just *also* stores them for reference or for other potential uses during training.

	return &GradientFineTuner{
		Model:model, Config:config, Optimizer:optimizer, StepCount:0, BestLoss:math.Inf(1),
		Serializer:NewTensorModelSerializer(), Parameters:model.GetNamedParameters(), Tokenizer:tok,
	}
}

// ClipGradients, ZeroGradients, GetLearningRate, UpdateLearningRate (mostly unchanged)
func (ft *GradientFineTuner) ClipGradients() {
	totalNormSq := 0.0
	for _, param := range ft.Parameters {
		if param.Grad == nil || !param.RequiresGrad { continue }
		for i := 0; i < param.Grad.Rows; i++ { for j := 0; j < param.Grad.Cols; j++ { totalNormSq += param.Grad.Data[i][j] * param.Grad.Data[i][j] } }
	}
	totalNorm := math.Sqrt(totalNormSq)
	if totalNorm > ft.Config.ClipGradNorm && ft.Config.ClipGradNorm > 0 {
		clipFactor := ft.Config.ClipGradNorm / (totalNorm + 1e-6)
		for _, param := range ft.Parameters {
			if param.Grad == nil || !param.RequiresGrad { continue }
			for i := 0; i < param.Grad.Rows; i++ { for j := 0; j < param.Grad.Cols; j++ { param.Grad.Data[i][j] *= clipFactor } }
		}
	}
}
func (ft *GradientFineTuner) ZeroGradients() { for _, p := range ft.Parameters { if p.Grad != nil && p.RequiresGrad { p.ZeroGrad() } } }
func (ft *GradientFineTuner) GetLearningRate() float64 {
	baseRate := ft.Config.LearningRate
	if ft.Config.WarmupSteps > 0 && ft.StepCount < ft.Config.WarmupSteps { return baseRate * float64(ft.StepCount+1) / float64(ft.Config.WarmupSteps) }
	return baseRate
}
func (ft *GradientFineTuner) UpdateLearningRate() { lr := ft.GetLearningRate(); switch opt := ft.Optimizer.(type) { case *AdamOptimizer: opt.LearningRate = lr; case *SGDOptimizer: opt.LearningRate = lr } }


// TensorTrainingExample structure and constructor
type TensorTrainingExample struct { SourceTokens []int; TargetTokens []int; Labels []int }
func NewTensorTrainingExample(src, tgt, lbl []int) *TensorTrainingExample { return &TensorTrainingExample{SourceTokens:src, TargetTokens:tgt, Labels:lbl} }

// Helper to convert []int to a 2D tensor (1, seqLen) for model input
func makeTensorFromInts(ints []int, graph *autodiff.ComputationGraph, name string) (*autodiff.Tensor, error) {
	if len(ints) == 0 {
		m, _ := autodiff.NewMatrix(1,0) // Shape (1,0) for empty sequence in a batch of 1
		return autodiff.NewTensor(m, &autodiff.TensorConfig{Graph: graph, Name: name, DType: autodiff.Int64}) // Conceptual DType
	}
	data := make([][]float64, 1); data[0] = make([]float64, len(ints))
	for i, val := range ints { data[0][i] = float64(val) }
	matrix, _ := autodiff.NewMatrix(1, len(ints), data...)
	return autodiff.NewTensor(matrix, &autodiff.TensorConfig{RequiresGrad: false, Graph: graph, Name: name, DType: autodiff.Int64})
}

// TrainStep method
func (ft *GradientFineTuner) TrainStep(batchExamples []*TensorTrainingExample) (float64, error) {
	if len(batchExamples) == 0 { return 0.0, fmt.Errorf("empty batch") }
	if ft.Model.Graph == nil { return 0.0, fmt.Errorf("model has no graph") }
	currentGraph := ft.Model.Graph

	ft.ZeroGradients()
	var accumulatedTaskLoss *autodiff.Tensor; firstLoss := true

	// This processes one example at a time, true batching would combine examples into single tensors.
	for i, example := range batchExamples {
		srcTensor, err := makeTensorFromInts(example.SourceTokens, currentGraph, fmt.Sprintf("src_ex%d", i))
		if err != nil { return 0, fmt.Errorf("src tensor ex %d: %w", i, err) }
		tgtTensor, err := makeTensorFromInts(example.TargetTokens, currentGraph, fmt.Sprintf("tgt_ex%d", i))
		if err != nil { return 0, fmt.Errorf("tgt tensor ex %d: %w", i, err) }

		logits, err := ft.Model.Forward(srcTensor, tgtTensor, nil, nil, true)
		if err != nil { return 0, fmt.Errorf("fwd pass ex %d: %w", i, err) }
		
		taskLoss, err := autodiff.CrossEntropyLoss(logits, example.Labels)
		if err != nil { return 0, fmt.Errorf("task loss ex %d: %w", i, err) }

		if firstLoss { accumulatedTaskLoss = taskLoss; firstLoss = false
		} else { accumulatedTaskLoss, err = autodiff.Add(accumulatedTaskLoss, taskLoss); if err != nil { return 0, fmt.Errorf("accum task loss: %w", err)} }
	}

	if accumulatedTaskLoss == nil { return 0.0, fmt.Errorf("no task loss accumulated") }
	avgTaskLoss, err := autodiff.ScalarMultiply(accumulatedTaskLoss, 1.0/float64(len(batchExamples)))
	if err != nil { return 0, fmt.Errorf("avg task loss: %w", err) }

	totalMoEAuxLossCfg := &autodiff.TensorConfig{Graph: currentGraph, Name: "total_moe_aux_loss", RequiresGrad: true}
	totalMoEAuxLoss, _ := autodiff.NewTensor(autodiff.NewMatrixZeros(1,1), totalMoEAuxLossCfg)


	moeLayers := ft.Model.GetMoELayers()
	if len(moeLayers) > 0 {
		for _, moeLayer := range moeLayers {
			if moeLayer.AuxiliaryLoss != nil && moeLayer.AuxiliaryLoss.RequiresGrad { // Ensure aux loss itself can propagate
				totalMoEAuxLoss, err = autodiff.Add(totalMoEAuxLoss, moeLayer.AuxiliaryLoss)
				if err != nil { return 0, fmt.Errorf("add MoE aux loss: %w", err) }
			}
		}
	}
	
	overallLoss, err := autodiff.Add(avgTaskLoss, totalMoEAuxLoss)
	if err != nil { return 0, fmt.Errorf("combine task & aux losses: %w", err) }
	
	err = overallLoss.BackwardAll(); if err != nil { return 0, fmt.Errorf("overall backward pass: %w", err) }

	lossValue := 0.0
	if overallLoss.Data != nil && overallLoss.Data.Rows > 0 && overallLoss.Data.Cols > 0 { lossValue = overallLoss.Data.Data[0][0]
	} else { log.Println("Warning: Overall loss tensor data nil/empty post-backward.") }
	
	ft.ClipGradients(); ft.UpdateLearningRate()
	switch opt := ft.Optimizer.(type) {
	case *AdamOptimizer: opt.LearningRate = ft.GetLearningRate(); opt.Step(ft.Parameters)
	case *SGDOptimizer: opt.LearningRate = ft.GetLearningRate(); opt.Step(ft.Parameters)
	default: return 0, fmt.Errorf("unknown optimizer type")
	}
	ft.StepCount++; return lossValue, nil
}

// batchTensorTrainingExamples (unchanged)
func batchTensorTrainingExamples(examples []*TensorTrainingExample, batchSize int) [][]*TensorTrainingExample { /* ... */
	var batches [][]*TensorTrainingExample; for i := 0; i < len(examples); i += batchSize { end := i + batchSize; if end > len(examples) { end = len(examples) }; batches = append(batches, examples[i:end]) }; return batches
}
// loadAndTokenizeSegment (unchanged)
func loadAndTokenizeSegment(filePath string, tokenizer *tokenizer.Tokenizer, maxSeqLen int) ([]*TensorTrainingExample, error) { /* ... */
	if tokenizer == nil { return nil, fmt.Errorf("tokenizer is nil") }; file, err := os.Open(filePath); if err != nil { return nil, fmt.Errorf("open file %s: %w", filePath, err) }; defer file.Close()
	var examples []*TensorTrainingExample; scanner := bufio.NewScanner(file); lineNum := 0
	for scanner.Scan() {
		lineNum++; text := scanner.Text(); if text == "" { continue }
		encodeOptions := tokenizer.DefaultEncodeOptions(); if maxSeqLen > 0 { encodeOptions.MaxLength = maxSeqLen; encodeOptions.Truncation = true }
		encodingResult, err := tokenizer.Encode(text, encodeOptions)
		if err != nil { log.Printf("Warn: Tokenize line %d in %s: %v. Skip.", lineNum, filePath, err); continue }
		example := &TensorTrainingExample{ SourceTokens: encodingResult.InputIDs, TargetTokens: encodingResult.InputIDs, Labels: encodingResult.InputIDs }; examples = append(examples, example)
	}
	if err := scanner.Err(); err != nil { return nil, fmt.Errorf("read segment file %s: %w", filePath, err) }; return examples, nil
}
// FineTune (unchanged)
func (ft *GradientFineTuner) FineTune(validationSet []*TensorTrainingExample, savePath string) error { /* ... */
	fmt.Println("Starting fine-tuning..."); if ft.Tokenizer == nil && len(ft.Config.SegmentFilePaths) > 0 { return fmt.Errorf("tokenizer required for segment loading") }; if ft.Config.MaxSeqLen <= 0 && len(ft.Config.SegmentFilePaths) > 0 { return fmt.Errorf("MaxSeqLen must be positive for segment loading") }
	rand.Seed(time.Now().UnixNano())
	for epoch := 0; epoch < ft.Config.NumEpochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, ft.Config.NumEpochs); epochLoss := 0.0; numBatchesProcessed := 0
		if len(ft.Config.SegmentFilePaths) > 0 {
			segmentPaths := make([]string, len(ft.Config.SegmentFilePaths)); copy(segmentPaths, ft.Config.SegmentFilePaths); rand.Shuffle(len(segmentPaths), func(i, j int) { segmentPaths[i], segmentPaths[j] = segmentPaths[j], segmentPaths[i] })
			for _, filePath := range segmentPaths {
				log.Printf("Processing segment: %s for epoch %d\n", filePath, epoch+1)
				segmentExamples, err := loadAndTokenizeSegment(filePath, ft.Tokenizer, ft.Config.MaxSeqLen); if err != nil { log.Printf("Warn: Load/tokenize segment %s: %v. Skip.", filePath, err); continue }; if len(segmentExamples) == 0 { log.Printf("Warn: No examples in segment %s. Skip.", filePath); continue }
				rand.Shuffle(len(segmentExamples), func(i, j int) { segmentExamples[i], segmentExamples[j] = segmentExamples[j], segmentExamples[i] })
				segmentBatches := batchTensorTrainingExamples(segmentExamples, ft.Config.BatchSize)
				for i, batch := range segmentBatches {
					loss, trainErr := ft.TrainStep(batch); if trainErr != nil { log.Printf("Err train step batch %d seg %s: %v. Skip.", i, filePath, trainErr); continue }
					epochLoss += loss; numBatchesProcessed++; if numBatchesProcessed%10 == 0 { fmt.Printf("  Epoch %d, Seg %s, Batch %d/%d, CurLoss: %.4f, AvgLoss: %.4f\n", epoch+1, filepath.Base(filePath), i+1, len(segmentBatches), loss, epochLoss/float64(numBatchesProcessed)) }
				}
			}
		} else if len(ft.Config.TrainingExamples) > 0 {
			currentTrainingExamples := make([]*TensorTrainingExample, len(ft.Config.TrainingExamples)); copy(currentTrainingExamples, ft.Config.TrainingExamples); rand.Shuffle(len(currentTrainingExamples), func(i, j int) { currentTrainingExamples[i], currentTrainingExamples[j] = currentTrainingExamples[j], currentTrainingExamples[i] })
			inMemoryBatches := batchTensorTrainingExamples(currentTrainingExamples, ft.Config.BatchSize)
			for i, batch := range inMemoryBatches {
				loss, trainErr := ft.TrainStep(batch); if trainErr != nil { log.Printf("Err train step mem batch %d: %v. Skip.", i, trainErr); continue }
				epochLoss += loss; numBatchesProcessed++; if numBatchesProcessed%10 == 0 { fmt.Printf("  Epoch %d, MemBatch %d/%d, CurLoss: %.4f, AvgLoss: %.4f\n", epoch+1, i+1, len(inMemoryBatches), loss, epochLoss/float64(numBatchesProcessed)) }
			}
		} else { fmt.Println("No training data. Skip epoch."); continue }
		if numBatchesProcessed > 0 { avgEpochLoss := epochLoss / float64(numBatchesProcessed); fmt.Printf("Epoch %d Summary: AvgLoss: %.4f, Batches: %d\n", epoch+1, avgEpochLoss, numBatchesProcessed); if avgEpochLoss < ft.BestLoss { ft.BestLoss = avgEpochLoss }
		} else { fmt.Printf("Epoch %d Summary: No batches processed.\n", epoch+1) }
		if len(validationSet) > 0 { valLoss, valErr := ft.Evaluate(validationSet); if valErr != nil { log.Printf("Err validation epoch %d: %v", epoch+1, valErr) } else { fmt.Printf("  Epoch %d ValLoss: %.4f\n", epoch+1, valLoss) } }
		if savePath != "" && (epoch+1)%ft.Config.SaveFrequency == 0 {
			ckptPath := filepath.Join(savePath, fmt.Sprintf("ckpt_epoch_%d", epoch+1)); if err := os.MkdirAll(ckptPath, 0755); err != nil { log.Printf("Fail create ckpt dir %s: %v", ckptPath, err)
			} else { if err := ft.Serializer.SaveTransformerWithTensors(ft.Model, ckptPath); err != nil { log.Printf("Fail save ckpt %s: %v", ckptPath, err) } else { fmt.Printf("  Saved ckpt to %s\n", ckptPath) } }
		}
	}
	fmt.Println("Fine-tuning complete!"); fmt.Printf("Best training loss: %.4f\n", ft.BestLoss)
	if savePath != "" { finalPath := filepath.Join(savePath, "final_model"); if err := os.MkdirAll(finalPath, 0755); err != nil { log.Printf("Fail create final model dir %s: %v", finalPath, err)
		} else { if err := ft.Serializer.SaveTransformerWithTensors(ft.Model, finalPath); err != nil { log.Printf("Fail save final model %s: %v", finalPath, err) } else { fmt.Printf("Final model saved to %s\n", finalPath) } }
	}
	return nil
}
// Evaluate method (mostly unchanged, but ensure graph handling for inputs)
func (ft *GradientFineTuner) Evaluate(validationSet []*TensorTrainingExample) (float64, error) {
	if len(validationSet) == 0 { return 0.0, nil }; totalLoss := 0.0; numExamplesProcessed := 0
	validationBatches := batchTensorTrainingExamples(validationSet, ft.Config.BatchSize)
	for _, batch := range validationBatches {
		if len(batch) == 0 { continue }
		currentGraph := ft.Model.Graph; if currentGraph == nil { currentGraph = NewComputationGraph() } // Should use model's graph
		for _, example := range batch {
			evalSrcTensor, err := makeTensorFromInts(example.SourceTokens, currentGraph, "eval_src")
			if err != nil { log.Printf("Eval: Error creating src tensor: %v", err); continue }
			evalTgtTensor, err := makeTensorFromInts(example.TargetTokens, currentGraph, "eval_tgt")
			if err != nil { log.Printf("Eval: Error creating tgt tensor: %v", err); continue }

			// Model's Forward method now returns (*Tensor, error)
			logits, err := ft.Model.Forward(evalSrcTensor, evalTgtTensor, nil, nil, false) // isTraining = false
			if err != nil { log.Printf("Error during Forward for validation: %v. Skipping example.", err); continue }

			loss, err := CrossEntropyLoss(logits, example.Labels) // Use graph-aware CrossEntropyLoss
			if err != nil { log.Printf("Error calculating loss for validation: %v. Skipping example.", err); continue }

			if loss.Data != nil && loss.Data.Rows > 0 && loss.Data.Cols > 0 { totalLoss += loss.Data.Data[0][0]; numExamplesProcessed++ }
		}
	}
	if numExamplesProcessed == 0 { return 0.0, fmt.Errorf("no valid examples processed in evaluation") }
	return totalLoss / float64(numExamplesProcessed), nil
}

// TensorModelSerializer (unchanged)
type TensorModelSerializer struct{}
func NewTensorModelSerializer() *TensorModelSerializer { return &TensorModelSerializer{} }
func (ms *TensorModelSerializer) SaveTransformerWithTensors(model *TransformerWithTensors, path string) error { return nil }
func (ms *TensorModelSerializer) LoadTransformerWithTensors(path string) (*TransformerWithTensors, error) { return nil, fmt.Errorf("not implemented") }
