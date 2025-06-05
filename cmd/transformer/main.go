package main

import (
	"fmt"
	"os"

	"transformer/pkg/autodiff"
	"transformer/pkg/core"
	"transformer/internal/tokenizer"
)

// Main entry point for the reorganized transformer library
func main() {
	fmt.Println("Transformer Library - Reorganized Structure")
	fmt.Println("==========================================")

	// Parse command line arguments
	mode := "default"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}

	switch mode {
	case "default":
		runDefaultExample()
	case "help":
		printHelp()
	default:
		fmt.Printf("Unknown mode: %s\n", mode)
		printHelp()
	}

	// TODO: Implement 'train' mode:
	// - Load training data
	// - Initialize optimizer
	// - Loop through epochs and batches:
	//   - Perform forward pass
	//   - Calculate loss
	//   - Perform backward pass (compute gradients)
	//   - Update model parameters
	//   - Save checkpoints

	// TODO: Implement 'inference' mode:
	// - Load pre-trained model (if available) or use the initialized one
	// - Take user input or load inference data
	// - Tokenize input
	// - Perform forward pass (potentially in a generation loop for text output)
	// - Decode output tokens to text
}

// runDefaultExample demonstrates basic transformer functionality
func runDefaultExample() {
	fmt.Println("\nRunning Default Example:")
	fmt.Println("------------------------")

	// Initialize configuration
	config := core.NewDefaultConfig()
	fmt.Println("Configuration initialized with:")
	fmt.Printf("- Vocabulary size: %d\n", config.VocabSize)
	fmt.Printf("- Embedding dimension: %d\n", config.EmbeddingDim)
	fmt.Printf("- Number of layers: %d\n", config.NumLayers)
	fmt.Printf("- Number of attention heads: %d\n", config.NumHeads)

	// Create a simple vocabulary for demonstration
	vocab := []string{"<UNK>", "hello", "world", "transformer", "model", "in", "go"}
	tokenizer := tokenizer.NewSimpleTokenizer(vocab)

	// Encode some text
	srcText := "hello world"
	fmt.Println("\nTokenizing text:", srcText)
	srcTokens := tokenizer.Encode(srcText)
	fmt.Println("Tokens:", srcTokens)

	// Initialize computation graph
	graph := autodiff.NewComputationGraph()

	// Initialize TransformerWithTensors
	transformerModel := autodiff.NewTransformerWithTensors(config, graph) // Renamed for clarity
	fmt.Println("\nTransformerWithTensors initialized.")

	// Define dummy source and target token sequences
	srcTokenData := [][]float64{{1, 2, 3}} // Batch size 1, sequence length 3
	tgtTokenData := [][]float64{{4, 5}}    // Batch size 1, sequence length 2
	fmt.Printf("\nDummy source tokens: %v\n", srcTokenData)
	fmt.Printf("Dummy target tokens: %v\n", tgtTokenData)

	// Convert token sequences into autodiff.Tensor objects
	srcMatrix := autodiff.NewMatrixFromFloat64(srcTokenData)
	srcTensor, err := autodiff.NewTensor(srcMatrix, &autodiff.TensorConfig{Graph: graph, RequiresGrad: false, Name: "src_tokens"})
	if err != nil {
		fmt.Printf("Error creating source tensor: %v\n", err)
		return
	}
	fmt.Printf("Source tensor created with shape: %v\n", srcTensor.Shape())

	tgtMatrix := autodiff.NewMatrixFromFloat64(tgtTokenData)
	tgtTensor, err := autodiff.NewTensor(tgtMatrix, &autodiff.TensorConfig{Graph: graph, RequiresGrad: false, Name: "tgt_tokens"})
	if err != nil {
		fmt.Printf("Error creating target tensor: %v\n", err)
		return
	}
	fmt.Printf("Target tensor created with shape: %v\n", tgtTensor.Shape())

	// Create dummy attention masks (using nil for now)
	fmt.Println("Using nil for source and target attention masks.")

	// Call the transformerModel.Forward() method
	fmt.Println("\nPerforming forward pass...")
	outputTensor, err := transformerModel.Forward(srcTensor, tgtTensor, nil, nil, false) // isTraining set to false

	if err != nil {
		fmt.Printf("Error during forward pass: %v\n", err)
	} else {
		fmt.Printf("Forward pass successful!\n")
		fmt.Printf("Output tensor shape: %v\n", outputTensor.Shape())
	}

	fmt.Println("\nTransformer library example with forward pass executed.")

	// TODO: Expand this example to showcase more detailed model interaction,
	// potentially including:
	// - Loading pre-trained weights (if applicable)
	// - A simple generation loop (e.g., autoregressive decoding for one token)
	// - Visualization of attention weights (if utilities are added)
}

// printHelp displays usage information
func printHelp() {
	fmt.Println("\nUsage: go run main.go [mode]")
	fmt.Println("\nAvailable modes:")
	fmt.Println("  default  - Run a simple transformer example")
	fmt.Println("  help     - Display this help message")
}
