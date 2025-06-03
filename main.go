package main

import (
	"fmt"
	"os"
	
	"github.com/transformer_reorganized/core"
	"github.com/transformer_reorganized/tokenizer"
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
	
	fmt.Println("\nTransformer library successfully initialized!")
}

// printHelp displays usage information
func printHelp() {
	fmt.Println("\nUsage: go run main.go [mode]")
	fmt.Println("\nAvailable modes:")
	fmt.Println("  default  - Run a simple transformer example")
	fmt.Println("  help     - Display this help message")
}
