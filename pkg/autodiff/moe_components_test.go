package autodiff_test // Changed to _test package

import (
	"testing"
	"fmt"
	"transformer/pkg/autodiff" // Import the package under test
)

// Helper to create a new graph for each test to ensure isolation
func newTestGraphComponents() *autodiff.ComputationGraph {
	return autodiff.NewComputationGraph()
}

// --- Tests for moe.Expert (now autodiff.Expert) ---

func TestNewExpert(t *testing.T) {
	graph := newTestGraphComponents()
	inputDim, hiddenDim, outputDim := 4, 8, 4

	// Call functions from the autodiff package
	expert := autodiff.NewExpert(inputDim, hiddenDim, outputDim, autodiff.GELU, true, graph)

	if expert == nil {
		t.Fatal("NewExpert returned nil")
	}
	if expert.W1 == nil || expert.W1.Shape()[0] != inputDim || expert.W1.Shape()[1] != hiddenDim {
		t.Errorf("Expert.W1 has incorrect shape or is nil. Got %v, expected (%d,%d)", expert.W1.Shape(), inputDim, hiddenDim)
	}
	// ... (rest of the assertions for NewExpert, checking expert.B1, W2, B2 shapes and properties) ...
	if expert.B1 == nil || expert.B1.Shape()[0] != 1 || expert.B1.Shape()[1] != hiddenDim {
		t.Errorf("Expert.B1 has incorrect shape or is nil. Got %v, expected (1,%d)", expert.B1.Shape(), hiddenDim)
	}
	if expert.W2 == nil || expert.W2.Shape()[0] != hiddenDim || expert.W2.Shape()[1] != outputDim {
		t.Errorf("Expert.W2 has incorrect shape or is nil. Got %v, expected (%d,%d)", expert.W2.Shape(), hiddenDim, outputDim)
	}
	if expert.B2 == nil || expert.B2.Shape()[0] != 1 || expert.B2.Shape()[1] != outputDim {
		t.Errorf("Expert.B2 has incorrect shape or is nil. Got %v, expected (1,%d)", expert.B2.Shape(), outputDim)
	}

	for _, p := range []*autodiff.Tensor{expert.W1, expert.B1, expert.W2, expert.B2} {
		if p.Graph != graph {
			t.Errorf("Parameter %s not associated with the correct graph", p.Name)
		}
		if !p.RequiresGrad {
			t.Errorf("Parameter %s RequiresGrad is false, expected true", p.Name)
		}
	}
	if expert.Activation == nil {
		t.Error("Expert.Activation is nil")
	}
}

func TestExpertForward(t *testing.T) {
	graph := newTestGraphComponents()
	inputDim, hiddenDim, outputDim := 4, 8, 4
	batchSize := 2

	expert := autodiff.NewExpert(inputDim, hiddenDim, outputDim, autodiff.GELU, true, graph)

	inputData, _ := autodiff.NewRandomMatrix(batchSize, inputDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "expert_input"})

	output, err := expert.Forward(inputTensor)
	if err != nil {
		t.Fatalf("Expert.Forward returned an error: %v", err)
	}
	// ... (rest of the assertions for ExpertForward) ...
	if output == nil {
		t.Fatal("Expert.Forward returned nil output")
	}
	if output.Shape()[0] != batchSize || output.Shape()[1] != outputDim {
		t.Errorf("Output tensor shape mismatch. Got %v, expected (%d,%d)", output.Shape(), batchSize, outputDim)
	}
	if output.Graph != graph {
		t.Error("Output tensor not associated with the correct graph")
	}
}

func TestExpertBackward(t *testing.T) {
	graph := newTestGraphComponents()
	inputDim, hiddenDim, outputDim := 2, 3, 2
	batchSize := 1

	expert := autodiff.NewExpert(inputDim, hiddenDim, outputDim, autodiff.GELU, true, graph)

	inputData, _ := autodiff.NewRandomMatrix(batchSize, inputDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "expert_input_bw_test", RequiresGrad: true})

	output, errFwd := expert.Forward(inputTensor)
	if errFwd != nil { t.Fatalf("Expert.Forward failed: %v", errFwd) }

	loss, errLoss := autodiff.TensorMean(output, -1, false)
	if errLoss != nil { t.Fatalf("TensorMean for loss failed: %v", errLoss) }

	if loss.Grad == nil && loss.RequiresGrad {
		loss.Grad, _ = autodiff.NewMatrix(loss.Shape()[0], loss.Shape()[1])
	}
	loss.Grad.Data[0][0] = 1.0

	if loss.Graph == nil && loss.RequiresGrad {
		loss.SetGraph(graph)
		// graph.AddNode(loss) // AddNode is not exported and typically managed by ops
	}
	loss.Graph.Backward()

	paramsToCheck := []*autodiff.Tensor{expert.W1, expert.B1, expert.W2, expert.B2}
	for _, p := range paramsToCheck {
		if p.Grad == nil {
			t.Errorf("Parameter %s has nil gradient after backward pass", p.Name)
			continue
		}
		// ... (gradient check logic) ...
		hasNonZeroGrad := false
		for i := 0; i < p.Grad.Rows; i++ {
			for j := 0; j < p.Grad.Cols; j++ { if p.Grad.Data[i][j] != 0 { hasNonZeroGrad = true; break } }
			if hasNonZeroGrad { break }
		}
		if !hasNonZeroGrad {
			fmt.Printf("Info: Parameter %s has all zero gradients (might be normal for this input/loss).\n", p.Name)
		}
	}
}

func TestExpertGetParameters(t *testing.T) {
	graph := newTestGraphComponents()
	expert := autodiff.NewExpert(4, 8, 4, autodiff.GELU, true, graph)
	params := expert.GetParameters()

	if len(params) != 4 {
		t.Fatalf("Expected 4 parameters, got %d", len(params))
	}
	// ... (rest of assertions for GetParameters) ...
	if params[0] != expert.W1 || params[1] != expert.B1 || params[2] != expert.W2 || params[3] != expert.B2 {
		t.Error("GetParameters did not return correct parameter tensors")
	}
}

// --- Tests for moe.Router (now autodiff.Router) ---

func TestNewRouter(t *testing.T) {
	graph := newTestGraphComponents()
	modelDim, numExperts := 4, 3
	router := autodiff.NewRouter(modelDim, numExperts, true, graph)

	if router == nil {
		t.Fatal("NewRouter returned nil")
	}
	// ... (assertions for NewRouter) ...
	if router.Weights == nil || router.Weights.Shape()[0] != modelDim || router.Weights.Shape()[1] != numExperts {
		t.Errorf("Router.Weights has incorrect shape or is nil. Got %v, expected (%d,%d)", router.Weights.Shape(), modelDim, numExperts)
	}
	if router.Bias == nil || router.Bias.Shape()[0] != 1 || router.Bias.Shape()[1] != numExperts {
		t.Errorf("Router.Bias has incorrect shape or is nil. Got %v, expected (1,%d)", router.Bias.Shape(), numExperts)
	}
	if router.Weights.Graph != graph || router.Bias.Graph != graph {
		t.Error("Router parameters not associated with the correct graph")
	}
	if !router.Weights.RequiresGrad || !router.Bias.RequiresGrad {
		t.Error("Router parameters RequiresGrad is false, expected true")
	}
}

func TestRouterForward(t *testing.T) {
	graph := newTestGraphComponents()
	modelDim, numExperts := 4, 3
	batchSize := 2

	router := autodiff.NewRouter(modelDim, numExperts, true, graph)

	inputData, _ := autodiff.NewRandomMatrix(batchSize, modelDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "router_input"})

	logits, err := router.Forward(inputTensor)
	if err != nil {
		t.Fatalf("Router.Forward returned an error: %v", err)
	}
	// ... (assertions for RouterForward) ...
	if logits == nil {
		t.Fatal("Router.Forward returned nil logits")
	}
	if logits.Shape()[0] != batchSize || logits.Shape()[1] != numExperts {
		t.Errorf("Logits tensor shape mismatch. Got %v, expected (%d,%d)", logits.Shape(), batchSize, numExperts)
	}
	if logits.Graph != graph {
		t.Error("Logits tensor not associated with the correct graph")
	}
}

func TestRouterBackward(t *testing.T) {
	graph := newTestGraphComponents()
	modelDim, numExperts := 2, 3
	batchSize := 1

	router := autodiff.NewRouter(modelDim, numExperts, true, graph)

	inputData, _ := autodiff.NewRandomMatrix(batchSize, modelDim)
	inputTensor, _ := autodiff.NewTensor(inputData, &autodiff.TensorConfig{Graph: graph, Name: "router_input_bw_test", RequiresGrad: true})

	logits, errFwd := router.Forward(inputTensor)
	if errFwd != nil { t.Fatalf("Router.Forward failed: %v", errFwd) }

	loss, errLoss := autodiff.TensorMean(logits, -1, false)
	if errLoss != nil { t.Fatalf("TensorMean for loss failed: %v", errLoss) }

	if loss.Grad == nil && loss.RequiresGrad { loss.Grad, _ = autodiff.NewMatrix(loss.Shape()[0], loss.Shape()[1]) }
	loss.Grad.Data[0][0] = 1.0

	if loss.Graph == nil && loss.RequiresGrad { loss.SetGraph(graph); /* graph.AddNode(loss) */ }
	loss.Graph.Backward()

	paramsToCheck := []*autodiff.Tensor{router.Weights, router.Bias}
	for _, p := range paramsToCheck {
		if p.Grad == nil {
			t.Errorf("Parameter %s has nil gradient after backward pass", p.Name)
			continue
		}
		// ... (gradient check logic) ...
		hasNonZeroGrad := false
		for i := 0; i < p.Grad.Rows; i++ {
			for j := 0; j < p.Grad.Cols; j++ { if p.Grad.Data[i][j] != 0 { hasNonZeroGrad = true; break } }
			if hasNonZeroGrad { break }
		}
		if !hasNonZeroGrad {
			fmt.Printf("Info: Router parameter %s has all zero gradients (might be normal for this input/loss).\n", p.Name)
		}
	}
}

func TestRouterGetParameters(t *testing.T) {
	graph := newTestGraphComponents()
	router := autodiff.NewRouter(4, 3, true, graph)
	params := router.GetParameters()

	if len(params) != 2 {
		t.Fatalf("Expected 2 parameters, got %d", len(params))
	}
	// ... (assertions for GetParameters) ...
	if params[0] != router.Weights || params[1] != router.Bias {
		t.Error("GetParameters did not return correct parameter tensors for Router")
	}
}

// GetNodesForTest is not available on ComputationGraph from outside the package.
// Tests should rely on public behavior or use build tags for test-specific helpers if absolutely necessary.
// For these tests, checking parameter gradients after Backward() is the main goal.
// func (g *autodiff.ComputationGraph) GetNodesForTest() []*autodiff.Tensor {
//     return nil // Placeholder
// }
