package autodiff_test // Changed to _test package

import (
	"testing"
	"fmt"
	// "math" // Not strictly needed for these tests yet

	"transformer/pkg/autodiff" // Import the package under test
)

// Helper to create a new graph for each test to ensure isolation
func newTestGraphMoELayer() *autodiff.ComputationGraph {
	return autodiff.NewComputationGraph()
}

// Helper to create a default MoELayerConfig for tests
// Note: MoELayerConfig itself is now part of the autodiff package.
func defaultTestMoELayerConfig(graph *autodiff.ComputationGraph) autodiff.MoELayerConfig {
	return autodiff.MoELayerConfig{ // Use autodiff.MoELayerConfig
		ModelDim:             4,
		NumExperts:           2,
		HiddenDim:            8, // Expert FFN hidden dimension
		TopK:                 1,
		CapacityFactor:       1.25,
		NoisyRouting:         false, // Keep false for simplicity in basic tests
		RouterZLossCoeff:     0.01,
		LoadBalanceLossCoeff: 0.01,
		Activation:           autodiff.GELU, // Use autodiff.GELU
	}
}

// --- Tests for moe.MoELayer (now autodiff.MoELayer) ---

func TestNewMoELayer(t *testing.T) {
	graph := newTestGraphMoELayer()
	config := defaultTestMoELayerConfig(graph)
	config.NumExperts = 3

	layer := autodiff.NewMoELayer(config, true, graph) // Use autodiff.NewMoELayer

	if layer == nil {
		t.Fatal("NewMoELayer returned nil")
	}
	// ... (rest of assertions for NewMoELayer, assuming Router and Experts are accessible if needed for checks) ...
	if layer.Router == nil { // Router is a field of MoELayer
		t.Error("MoELayer.Router is nil")
	}
	if len(layer.Experts) != config.NumExperts { // Experts is a field
		t.Errorf("Expected %d experts, got %d", config.NumExperts, len(layer.Experts))
	}
	if layer.Experts[0] == nil {
		t.Error("MoELayer.Experts[0] is nil")
	}
	// Ensure internal components are on the graph (assuming these fields are accessible for testing)
	// This might require exporting fields or having specific test helpers in 'autodiff' package
	// For now, we assume direct field access for simplicity in test.
	if layer.Router.Weights.Graph != graph {
		t.Error("Router weights not on the correct graph")
	}
	if layer.Experts[0].W1.Graph != graph { // W1 is a field of Expert
		t.Error("Expert0.W1 not on the correct graph")
	}
	if layer.Config.NumExperts != config.NumExperts { // Config is a field
		t.Errorf("Config not stored correctly: NumExperts mismatch")
	}
	if layer.AuxiliaryLoss == nil { // AuxiliaryLoss is a field
		t.Error("MoELayer.AuxiliaryLoss is nil after initialization")
	}
	if layer.AuxiliaryLoss.Graph != graph {
		t.Error("Initial AuxiliaryLoss not on the correct graph")
	}
}

func TestMoELayerForward_TopK1(t *testing.T) {
	graph := newTestGraphMoELayer()
	config := defaultTestMoELayerConfig(graph)
	config.NumExperts = 2

	layer := autodiff.NewMoELayer(config, true, graph)

	batchSize := 3
	inputData, _ := autodiff.NewRandomMatrix(batchSize, config.ModelDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "moe_layer_input"})

	output, err := layer.Forward(inputTensor, true)

	if err != nil {
		t.Fatalf("MoELayer.Forward (TopK=1) returned an error: %v", err)
	}
	// ... (rest of assertions for Forward TopK1) ...
	if output == nil {
		t.Fatal("MoELayer.Forward (TopK=1) returned nil output")
	}
	expectedShape := []int{batchSize, config.ModelDim}
	if output.Shape()[0] != expectedShape[0] || output.Shape()[1] != expectedShape[1] {
		t.Errorf("Output tensor shape mismatch. Got %v, expected %v", output.Shape(), expectedShape)
	}
	if output.Graph != graph {
		t.Error("Output tensor not associated with the correct graph for TopK=1")
	}
	if layer.AuxiliaryLoss == nil {
		t.Fatal("MoELayer.AuxiliaryLoss is nil after forward pass (TopK=1, isTraining=true)")
	}
	if layer.AuxiliaryLoss.Shape()[0] != 1 || layer.AuxiliaryLoss.Shape()[1] != 1 {
		t.Errorf("AuxiliaryLoss shape is not scalar-like. Got %v", layer.AuxiliaryLoss.Shape())
	}
	fmt.Printf("TopK=1 Aux Loss value: %f (coeffs: RZ=%f, LB=%f)\n", layer.AuxiliaryLoss.Data.Data[0][0], config.RouterZLossCoeff, config.LoadBalanceLossCoeff)
}

func TestMoELayerForward_TopK2(t *testing.T) {
	graph := newTestGraphMoELayer()
	config := defaultTestMoELayerConfig(graph)
	config.TopK = 2
	config.NumExperts = 3

	layer := autodiff.NewMoELayer(config, true, graph)

	batchSize := 3
	inputData, _ := autodiff.NewRandomMatrix(batchSize, config.ModelDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "moe_layer_input_topk2"})

	output, err := layer.Forward(inputTensor, true)

	if err != nil {
		t.Logf("MoELayer.Forward (TopK=2) returned an error: %v", err)
	}
	// ... (rest of assertions for Forward TopK2) ...
	if output == nil {
		t.Fatal("MoELayer.Forward (TopK=2) returned nil output")
	}
	expectedShape := []int{batchSize, config.ModelDim}
	if output.Shape()[0] != expectedShape[0] || output.Shape()[1] != expectedShape[1] {
		t.Errorf("Output tensor shape mismatch (TopK=2). Got %v, expected %v", output.Shape(), expectedShape)
	}
	if output.Graph != graph {
		t.Error("Output tensor not associated with the correct graph for TopK=2")
	}
	if layer.AuxiliaryLoss == nil {
		t.Fatal("MoELayer.AuxiliaryLoss is nil after forward pass (TopK=2, isTraining=true)")
	}
	fmt.Printf("TopK=2 Aux Loss value: %f (coeffs: RZ=%f, LB=%f)\n", layer.AuxiliaryLoss.Data.Data[0][0], config.RouterZLossCoeff, config.LoadBalanceLossCoeff)
}

func TestMoELayerBackward_TopK1(t *testing.T) {
	graph := newTestGraphMoELayer()
	config := defaultTestMoELayerConfig(graph)
	config.NumExperts = 2

	layer := autodiff.NewMoELayer(config, true, graph)

	batchSize := 1
	inputData, _ := autodiff.NewRandomMatrix(batchSize, config.ModelDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "moe_layer_input_bw", RequiresGrad: true})

	output, errFwd := layer.Forward(inputTensor, true)
	if errFwd != nil { t.Fatalf("MoELayer.Forward (TopK=1) for backward test failed: %v", errFwd) }

	taskLoss, errLoss := autodiff.TensorMean(output, -1, false)
	if errLoss != nil { t.Fatalf("TensorMean for taskLoss failed: %v", errLoss) }

	totalLoss, errAdd := autodiff.Add(taskLoss, layer.AuxiliaryLoss)
	if errAdd != nil { t.Fatalf("Adding task and auxiliary loss failed: %v", errAdd) }

	if totalLoss.Grad == nil && totalLoss.RequiresGrad { totalLoss.Grad, _ = autodiff.NewMatrix(1,1) }
	totalLoss.Grad.Data[0][0] = 1.0

	if totalLoss.Graph == nil && totalLoss.RequiresGrad {
		totalLoss.SetGraph(graph)
		// graph.AddNode(totalLoss) // AddNode is not exported
	}
	totalLoss.Graph.Backward()

	for _, p := range layer.Router.GetParameters() {
		if p.Grad == nil { t.Errorf("Router parameter %s has nil gradient", p.Name); continue }
	}
	for i, expert := range layer.Experts {
		for _, p := range expert.GetParameters() {
			if p.Grad == nil { t.Errorf("Expert %d parameter %s has nil gradient", i, p.Name); continue }
		}
	}
}

func TestMoELayerGetParameters(t *testing.T) {
	graph := newTestGraphMoELayer()
	config := defaultTestMoELayerConfig(graph)
	config.NumExperts = 2
	layer := autodiff.NewMoELayer(config, true, graph)

	params := layer.GetParameters()
	expectedNumParams := len(layer.Router.GetParameters()) + config.NumExperts * len(layer.Experts[0].GetParameters())

	if len(params) != expectedNumParams {
		t.Fatalf("Expected %d parameters, got %d", expectedNumParams, len(params))
	}
	// ... (rest of assertions for GetParameters) ...
	isRouterWeight := false; isExpert0W1 := false
	for _, p := range params {
		if p == layer.Router.Weights { isRouterWeight = true }
		if p == layer.Experts[0].W1 { isExpert0W1 = true }
	}
	if !isRouterWeight { t.Error("Router.Weights not found in GetParameters result") }
	if !isExpert0W1 { t.Error("Experts[0].W1 not found in GetParameters result") }
}

func TestMoELayerForward_IsTrainingFalse(t *testing.T) {
	graph := newTestGraphMoELayer()
	config := defaultTestMoELayerConfig(graph)
	layer := autodiff.NewMoELayer(config, false, graph)

	batchSize := 3
	inputData, _ := autodiff.NewRandomMatrix(batchSize, config.ModelDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "moe_input_eval"})

	_, err := layer.Forward(inputTensor, false)
	if err != nil {
		t.Fatalf("MoELayer.Forward (isTraining=false) returned an error: %v", err)
	}

	if layer.AuxiliaryLoss == nil {
		t.Fatal("MoELayer.AuxiliaryLoss is nil after forward pass (isTraining=false)")
	}
	if layer.AuxiliaryLoss.Data.Data[0][0] != 0.0 {
		t.Errorf("Expected AuxiliaryLoss to be 0.0 when isTraining=false, got %f", layer.AuxiliaryLoss.Data.Data[0][0])
	}
}
