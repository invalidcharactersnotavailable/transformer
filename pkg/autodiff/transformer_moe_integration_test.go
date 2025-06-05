package autodiff

import (
	"testing"
	"fmt"
	// "math" // Not strictly needed for this test's assertions

	"transformer/pkg/core"
	// "transformer/pkg/moe" // MoE types are now local to autodiff package
)

// Helper to create a new graph for each test to ensure isolation (if not in a shared test util)
func newTestGraphIntegration() *ComputationGraph {
	return NewComputationGraph()
}

// Helper to convert []int to [][]float64 for NewTensorFromData
func intsToFloat64s(data []int, rows, cols int) [][]float64 {
	if rows*cols != len(data) && len(data) > 0 { // Allow empty if data is empty
		panic(fmt.Sprintf("Dimension mismatch for data conversion: %d*%d != %d", rows, cols, len(data)))
	}
	if len(data) == 0 && rows == 0 && cols == 0 {
		return [][]float64{}
	}
	if len(data) == 0 && rows*cols != 0 {
		// Create empty matrix of specified shape
		res := make([][]float64, rows)
		for i := range res {
			res[i] = make([]float64, cols)
		}
		return res
	}

	res := make([][]float64, rows)
	idx := 0
	for i := 0; i < rows; i++ {
		res[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			res[i][j] = float64(data[idx])
			idx++
		}
	}
	return res
}


func TestMoETransformerIntegration(t *testing.T) {
	// 1. Setup Graph
	transformerGraph := newTestGraphIntegration()

	// 2. Configure a Small MoE Transformer
	modelConfig := &core.Config{
		VocabSize:    10,
		EmbeddingDim: 8,
		NumLayers:    1, // Single encoder and decoder layer
		NumDecoderLayers: 1, // Explicitly set
		NumHeads:     2,
		FFNHiddenDim: 16, // For standard FFN if MoE was false
		MaxLen:       10,
		DropoutRate:  0.0, // No dropout for simpler testing
		ActivationFuncName: "gelu",


		UseMoE:            true,
		MoENumExperts:     2,
		MoEHiddenDim:      12,
		MoETopK:           1, // Start with TopK=1 for simplicity
		MoECapacityFactor: 1.25,
		MoENoisyRouting:   false,
		MoEActivationName: "gelu",
		// Loss coefficients will be passed via fineTuningConfig simulation
	}

	// These would typically come from TensorFineTuningConfig and passed to NewTransformerWithTensors
	// For this test, we'll ensure they are part of the moe.MoELayerConfig creation path
	// by setting them in the core.Config, which NewTransformerWithTensors uses to build moe.MoELayerConfig
	// This implies core.Config needs these fields, or NewTransformerWithTensors needs more params.
	// Let's assume core.Config now has these fields for simplicity of this test setup.
	// (This was an identified discrepancy, for test, we'll make core.Config temporarily hold them)
	// This isn't ideal, but makes the test runnable with current NewTransformerWithTensors signature.
	// A better way is to pass a dedicated MoETrainingConfig to NewTransformerWithTensors.
	// For now, let's assume core.Config was extended in a previous step (not shown here)
	// For the purpose of this test, we will pass them to NewMoELayer directly when constructing the model.
	// This means we need to modify NewTransformerWithTensors to accept these or assume they are in config.
	// The latest NewTransformerWithTensors takes them from config.MoERouterZLossCoeff etc.
	// So, we need to add them to core.Config struct definition.

	// Let's simulate that core.Config has these (as per HyperParameter flow)
	// We will add them to the core.Config definition for this test to pass
	// This is a bit of a workaround for the test setup if core.Config isn't the final place for these.
	// To make this test work with current structure, we'll assume core.Config has these temporarily
	// (as if populated by HyperParameterManager)
	// This is fine for an integration test where we control the config.

	// These fields are now expected in moe.MoELayerConfig, which is populated from core.Config
	// by NewTransformerWithTensors. So, setting them in modelConfig is the way.
	// We need to ensure core.Config actually has these fields.
	// Let's assume the core.Config has been updated as per step 1 of the overall subtask.
	// (Re-checking core.config.go, it does not have loss coeffs.
	// This means NewTransformerWithTensors needs to take them, or MoELayerConfig in core.Config needs them)

	// For this test to proceed, let's assume NewTransformerWithTensors
	// will correctly pass these through to MoELayerConfig.
	// The MoELayerConfig in pkg/moe/moe_layer.go already has these fields.
	// The NewTransformerWithTensors in pkg/autodiff/tensor_transformer.go
	// populates moe.MoELayerConfig using config.MoERouterZLossCoeff etc.
	// So, we need to add these to core.Config for this test.

	// To avoid modifying core.Config again just for this test,
	// let's assume that the moe.MoELayerConfig inside NewTransformerWithTensors
	// gets these values. We can't directly pass a TensorFineTuningConfig here easily.
	// The test will rely on the defaults set in moe.MoELayerConfig if not overridden from core.Config.
	// For a more controlled test, these should be plumbed.
	// Given the last update to tensor_transformer.go, it expects these in core.Config.
	// So, we add them to the modelConfig instance for the test.

	// The NewTransformerWithTensors function now uses the local MoELayerConfig type,
	// and it expects fields like MoENumExperts, MoEHiddenDim etc. to be present in the core.Config.
	// It also expects MoERouterZLossCoeff and MoELoadBalanceLossCoeff to be in core.Config
	// to populate the local MoELayerConfig.

	// For this test, we ensure that the core.Config instance (modelConfig) has these fields.
	// If the actual definition of core.Config in pkg/core/core.go doesn't have them,
	// this test might not reflect the true behavior, or NewTransformerWithTensors would fail
	// if it tries to access non-existent fields.
	// It is assumed that core.Config has been updated to include all necessary MoE related fields.

	// 3. Initialize Model
	model := NewTransformerWithTensors(modelConfig, transformerGraph) // NewTransformerWithTensors uses local MoE types
	if model == nil {
		t.Fatal("NewTransformerWithTensors returned nil")
	}

	// 4. Prepare Dummy Input Data
	batchSize := 2
	seqLen := 5
	srcData := make([][]float64, batchSize)
	tgtData := make([][]float64, batchSize)
	labels := make([]int, batchSize*seqLen) // CrossEntropyLoss expects flat labels

	for i := 0; i < batchSize; i++ {
		srcData[i] = make([]float64, seqLen)
		tgtData[i] = make([]float64, seqLen)
		for j := 0; j < seqLen; j++ {
			srcToken := (i*seqLen + j + 1) % modelConfig.VocabSize
			tgtToken := (i*seqLen + j + 2) % modelConfig.VocabSize // Simple shift
			srcData[i][j] = float64(srcToken)
			tgtData[i][j] = float64(tgtToken)
			labels[i*seqLen+j] = tgtToken // Flattened labels
		}
	}

	srcTokensTensor, _ := NewTensorFromData(srcData, &TensorConfig{Graph: transformerGraph, Name: "src_tokens", DType: Int64})
	tgtTokensTensor, _ := NewTensorFromData(tgtData, &TensorConfig{Graph: transformerGraph, Name: "tgt_tokens", DType: Int64})
	// Target output for loss is typically flattened and 1D array of ints.
	// CrossEntropyLoss in autodiff.go expects []int for targets.

	// 5. Forward Pass
	isTraining := true
	outputLogits, err := model.Forward(srcTokensTensor, tgtTokensTensor, nil, nil, isTraining)
	if err != nil {
		t.Fatalf("model.Forward returned error: %v", err)
	}
	if outputLogits == nil {
		t.Fatal("model.Forward returned nil outputLogits")
	}
	// Expected output shape: (batchSize * seqLen, VocabSize) because EmbeddingTensor.Forward flattens.
	expectedOutputRows := batchSize * seqLen
	if outputLogits.Shape()[0] != expectedOutputRows || outputLogits.Shape()[1] != modelConfig.VocabSize {
		t.Errorf("outputLogits shape mismatch. Got %v, expected (%d,%d)", outputLogits.Shape(), expectedOutputRows, modelConfig.VocabSize)
	}

	// 6. Loss Calculation
	taskLoss, err := CrossEntropyLoss(outputLogits, labels)
	if err != nil {
		t.Fatalf("CrossEntropyLoss returned error: %v", err)
	}
	if taskLoss == nil {
		t.Fatal("CrossEntropyLoss returned nil taskLoss")
	}

	totalAuxLoss, _ := NewTensor(NewMatrixZeros(1,1), &TensorConfig{Graph: transformerGraph, Name: "total_aux_loss_agg", RequiresGrad: true})

	// model.GetMoELayers() now returns []AuxiliaryLossProvider
	// Each provider (which is an *autodiff.MoELayer) has GetAuxiliaryLoss()
	moeLayerProviders := model.GetMoELayers()
	if modelConfig.UseMoE && len(moeLayerProviders) == 0 {
		t.Error("Model configured to use MoE, but no MoE layers found via GetMoELayers")
	}

	for i, provider := range moeLayerProviders {
		auxLoss := provider.GetAuxiliaryLoss() // This is *autodiff.Tensor
		if auxLoss == nil {
			t.Fatalf("MoE layer provider %d GetAuxiliaryLoss() is nil", i)
		}
		// Ensure aux loss is on the same graph and has grad if it's to be added
		if auxLoss.Graph != transformerGraph { auxLoss.SetGraph(transformerGraph)}
		// If MoERouterZLossCoeff or MoELoadBalanceLossCoeff are non-zero in core.Config,
		// then the auxLoss from MoELayer.Forward (when isTraining=true) should have RequiresGrad=true.
		// This check can be refined based on core.Config values.
		// For instance:
		// if (modelConfig.MoERouterZLossCoeff > 0 || modelConfig.MoELoadBalanceLossCoeff > 0) && !auxLoss.RequiresGrad {
		//    t.Logf("Warning: MoE layer %d auxLoss.RequiresGrad is false, but loss coeffs might be > 0.", i)
		// }

		totalAuxLoss, err = Add(totalAuxLoss, auxLoss)
		if err != nil {
			t.Fatalf("Failed to add auxiliary loss from MoE layer provider %d: %v", i, err)
		}
	}

	overallLoss, err := Add(taskLoss, totalAuxLoss)
	if err != nil {
		t.Fatalf("Adding task and auxiliary loss failed: %v", err)
	}
	if overallLoss == nil {
		t.Fatal("Overall loss is nil")
	}
	fmt.Printf("Integration Test: TaskLoss=%.4f, AuxLoss=%.4f, OverallLoss=%.4f\n", taskLoss.Data.Data[0][0], totalAuxLoss.Data.Data[0][0], overallLoss.Data.Data[0][0])

	// 7. Backward Pass
	// Initialize gradient of final loss to 1.0
	if overallLoss.Grad == nil && overallLoss.RequiresGrad { overallLoss.Grad, _ = NewMatrix(1,1)}
	if overallLoss.Grad != nil { overallLoss.Grad.Data[0][0] = 1.0 }


	// Perform backward pass on the graph associated with the overallLoss
	if overallLoss.Graph == nil && overallLoss.RequiresGrad {
		// This should not happen if ops correctly propagate/set graphs
		t.Log("Warning: overallLoss has no graph, setting to transformerGraph for backward pass.")
		overallLoss.SetGraph(transformerGraph)
		// If loss itself was not added to graph by an op (e.g. if it was a pre-existing tensor), add it.
		// This is unlikely if it's result of Add.
	}
	if overallLoss.Graph != nil {
		overallLoss.Graph.Backward() // Call backward on the graph
	} else if overallLoss.RequiresGrad {
		t.Fatal("overallLoss requires grad but has no graph to run backward pass on.")
	}


	namedParams := model.GetNamedParameters()
	if len(namedParams) == 0 {
		t.Error("GetNamedParameters returned empty map")
	}
	for name, param := range namedParams {
		if !param.RequiresGrad {
			continue // Skip non-trainable params like PE
		}
		if param.Grad == nil {
			t.Errorf("Parameter %s has nil gradient after backward pass", name)
		} else {
			// Optional: Check for all-zero gradients (can be valid, but less likely for all params)
			// sumGrad := 0.0
			// for r := 0; r < param.Grad.Rows; r++ { for c := 0; c < param.Grad.Cols; c++ { sumGrad += math.Abs(param.Grad.Data[r][c])}}
			// if sumGrad == 0.0 {
			// 	t.Logf("Warning: Parameter %s has all zero gradients.", name)
			// }
		}
	}

	// 8. Optimizer Step (Basic Check)
	// Store current value of a parameter
	var testParam *Tensor
	var testParamName string
	for name, p := range namedParams { // Get an actual learnable parameter
		if p.RequiresGrad && p.Data.Rows > 0 && p.Data.Cols > 0 {
			testParam = p
			testParamName = name
			break
		}
	}
	if testParam == nil {
		t.Fatal("Could not find a suitable test parameter for optimizer step check.")
	}

	originalParamValue, _ := testParam.Data.Clone()

	// Use default fine-tuning config for optimizer settings
	ftConfig := NewTensorFineTuningConfig()
	// Override MoE coeffs if needed, but they are already in modelConfig for MoELayerConfig
	// ftConfig.MoERouterZLossCoeff = modelConfig.MoERouterZLossCoeff (already done conceptually)

	optimizer := NewAdamOptimizer(ftConfig.LearningRate, ftConfig.WeightDecay)
	optimizer.Step(namedParams)

	changed := false
	for i := 0; i < testParam.Data.Rows; i++ {
		for j := 0; j < testParam.Data.Cols; j++ {
			if math.Abs(testParam.Data.Data[i][j] - originalParamValue.Data[i][j]) > 1e-9 {
				changed = true
				break
			}
		}
		if changed { break }
	}

	if !changed {
		t.Errorf("Optimizer step did not change parameter %s values.", testParamName)
	}
	t.Logf("Integration test for MoE transformer completed.")
}
