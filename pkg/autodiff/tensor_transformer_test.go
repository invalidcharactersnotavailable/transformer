package autodiff

import (
	"testing"
	coreconfig "github.com/transformer_reorganized/pkg/core" // Alias for core config
)

func TestParameterSharingReducesParameterCount(t *testing.T) {
	// Configuration for the models
	config := &coreconfig.Config{
		VocabSize:    1000,
		EmbeddingDim: 64,
		NumLayers:    4, // Using 4 layers to clearly see the effect
		NumHeads:     4,
		FFNHiddenDim: 128,
		MaxLen:       50,
		UseCrossLayerParameterSharing: false,
	}

	// Model 1: No parameter sharing
	model1 := NewTransformerWithTensors(config)
	params1 := model1.GetParameters()
	countParams1 := len(params1)
	t.Logf("Model 1 (No Sharing) Parameter Count: %d", countParams1)

	// Model 2: With parameter sharing
	config.UseCrossLayerParameterSharing = true
	model2 := NewTransformerWithTensors(config)
	params2 := model2.GetParameters()
	countParams2 := len(params2)
	t.Logf("Model 2 (With Sharing) Parameter Count: %d", countParams2)

	// Assertions
	if countParams2 >= countParams1 {
		t.Errorf("Expected parameter count with sharing (%d) to be less than without sharing (%d)", countParams2, countParams1)
	}

	// More specific assertions (conceptual, as exact numbers depend on GetParameters naming)
	// For encoder layers (assuming at least one encoder layer exists)
	if config.NumLayers > 0 {
		// Expected params for one encoder layer:
		// SelfAttention (4 weights) + Norm1 (2 weights) + Norm2 (2 weights) + FFN (4 weights) = 12 tensors
		// Expected params for one decoder layer (if symmetric and has cross-attn):
		// SelfAttention (4) + CrossAttention (4) + Norm1(2) + Norm2(2) + Norm3(2) + FFN(4) = 18 tensors

		// Non-shared: NumLayers * 12 (encoder) + NumLayers * 18 (decoder) + embedding + output
		// Shared: 12 (encoder) + 18 (decoder) + embedding + output

		// Count non-embedding/output parameters
		nonSharedSpecificParams1 := 0
		for name := range params1 {
			if name != "embedding_matrix" && name != "output_matrix" {
				nonSharedSpecificParams1++
			}
		}

		nonSharedSpecificParams2 := 0
		for name := range params2 {
			if name != "embedding_matrix" && name != "output_matrix" {
				nonSharedSpecificParams2++
			}
		}

		t.Logf("Model 1 (No Sharing) Encoder/Decoder Layer Parameter Groups: %d", nonSharedSpecificParams1)
		t.Logf("Model 2 (With Sharing) Encoder/Decoder Layer Parameter Groups: %d", nonSharedSpecificParams2)


		// If NumLayers = 4, nonSharedSpecificParams1 should be approx 4 * (params_per_enc_layer + params_per_dec_layer)
		// nonSharedSpecificParams2 should be approx (params_per_enc_layer + params_per_dec_layer)
		// This means nonSharedSpecificParams2 should be roughly nonSharedSpecificParams1 / NumLayers.
		// This is an approximation because the naming in GetParameters might group them differently.
		// A more robust check would be to count distinct Tensor pointers for layer weights.

		// For simplicity, we rely on the overall count reduction.
		// A detailed check would require knowing the exact number of parameters per layer.
		// For encoder only (if decoder is also NumLayers):
		// Expected non-shared layer params = NumLayers * (4 SA + 2 LN1 + 2 LN2 + 4 FFN) = NumLayers * 12
		// Expected shared layer params = 12
		// So, if only encoders: countParams2 should be countParams1 - (NumLayers-1)*12 (encoder params)
		// If model has both encoder and decoder stacks of NumLayers each:
		// Expected non-shared layer params = NumLayers * (12_enc + 18_dec) = NumLayers * 30
		// Expected shared layer params = 12_enc + 18_dec = 30
		// So, countParams2 should be approx countParams1 - (NumLayers-1)*30

		// Example: 4 layers, encoder and decoder.
		// Params per enc layer: 4 (attn) + 4 (ffn) + 4 (ln_gamma/beta * 2) = 12
		// Params per dec layer: 8 (attn) + 4 (ffn) + 6 (ln_gamma/beta * 3) = 18
		// Total layer params (no sharing, 4 layers each): 4 * 12 (enc) + 4 * 18 (dec) = 48 + 72 = 120
		// Total layer params (sharing, 4 layers each): 1 * 12 (enc) + 1 * 18 (dec) = 30
		// Difference should be 120 - 30 = 90
		// So, countParams1 - countParams2 should be approx (NumLayers - 1) * (params_per_enc_layer + params_per_dec_layer)

		// Let's check that params2 has "shared_encoder_..." and "shared_decoder_..." keys
		// and not "encoder_1_..." if NumLayers > 1
		hasSharedEncoderKey := false
		hasIndexedEncoderKey := false
		hasSharedDecoderKey := false
		hasIndexedDecoderKey := false

		for name := range params2 {
			if name == "shared_encoder_self_query" { hasSharedEncoderKey = true }
			if name == "encoder_1_self_query" { hasIndexedEncoderKey = true } // Assuming NumLayers > 1
			if name == "shared_decoder_self_query" { hasSharedDecoderKey = true }
			if name == "decoder_1_self_query" { hasIndexedDecoderKey = true}
		}

		if config.NumLayers > 1 {
			if !hasSharedEncoderKey {
				t.Errorf("Expected shared encoder parameter keys when sharing is enabled and NumLayers > 0, but not found.")
			}
			if hasIndexedEncoderKey {
				t.Errorf("Found indexed encoder parameter key 'encoder_1_self_query' when sharing should be active.")
			}
			if !hasSharedDecoderKey { // This assumes decoder layers also exist and are shared
				t.Errorf("Expected shared decoder parameter keys when sharing is enabled and NumLayers > 0, but not found.")
			}
			if hasIndexedDecoderKey {
				t.Errorf("Found indexed decoder parameter key 'decoder_1_self_query' when sharing should be active.")
			}
		}
	}
}

// Mocking a simple tokenizer for finetuning tests
type SimpleTestTokenizer struct {
	Vocab map[string]int
	IDToToken map[int]string
	UnkID int
	PadID int
	MaxLen int
}

func NewSimpleTestTokenizer(maxLen int) *SimpleTestTokenizer {
	vocab := map[string]int{"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3, "test": 4, "data":5, ".":6}
	idToToken := make(map[int]string)
	for k, v := range vocab {
		idToToken[v] = k
	}
	return &SimpleTestTokenizer{
		Vocab: vocab,
		IDToToken: idToToken,
		UnkID: 1,
		PadID: 0,
		MaxLen: maxLen,
	}
}

func (s *SimpleTestTokenizer) Encode(text string, options *tokenizer.EncodeOptions) (*tokenizer.EncodingResult, error) {
	words := strings.Fields(strings.ToLower(text))
	ids := []int{}
	tokens := []string{}

	for _, word := range words {
		id, ok := s.Vocab[word]
		if !ok {
			id = s.UnkID
		}
		ids = append(ids, id)
		tokens = append(tokens, s.IDToToken[id])
	}

	maxLength := s.MaxLen
	if options != nil && options.MaxLength > 0 {
		maxLength = options.MaxLength
	}

	if len(ids) > maxLength {
		if options != nil && options.TruncationSide == "left" {
			ids = ids[len(ids)-maxLength:]
			tokens = tokens[len(tokens)-maxLength:]
		} else {
			ids = ids[:maxLength]
			tokens = tokens[:maxLength]
		}
	}

	attentionMask := make([]int, len(ids))
	for i := range ids {
		attentionMask[i] = 1
	}

	// Simple padding for testing
	if options != nil && options.Padding && len(ids) < maxLength {
		padCount := maxLength - len(ids)
		for i := 0; i < padCount; i++ {
			if options != nil && options.PaddingSide == "left" {
				ids = append([]int{s.PadID}, ids...)
				tokens = append([]string{s.IDToToken[s.PadID]}, tokens...)
				attentionMask = append([]int{0}, attentionMask...)
			} else {
				ids = append(ids, s.PadID)
				tokens = append(tokens, s.IDToToken[s.PadID])
				attentionMask = append(attentionMask, 0)
			}
		}
	}


	return &tokenizer.EncodingResult{
		InputIDs:      ids,
		Tokens:        tokens,
		AttentionMask: attentionMask,
		TokenTypeIDs:  make([]int, len(ids)), // All zeros for single sequence
	}, nil
}

// Adding other required methods for tokenizer.Tokenizer interface with minimal implementation
func (s *SimpleTestTokenizer) Decode(ids []int, options *tokenizer.DecodeOptions) (string, error) {
    var tokens []string
    for _, id := range ids {
        token, ok := s.IDToToken[id]
        if !ok {
            token = s.IDToToken[s.UnkID]
        }
        // Basic skip special tokens
        if options != nil && options.SkipSpecialTokens && (id == s.PadID || id == s.UnkID) { // Add other special tokens if needed
            continue
        }
        tokens = append(tokens, token)
    }
    return strings.Join(tokens, " "), nil
}

func (s *SimpleTestTokenizer) GetVocabSize() int {
    return len(s.Vocab)
}

func (s *SimpleTestTokenizer) DefaultEncodeOptions() *tokenizer.EncodeOptions {
	return &tokenizer.EncodeOptions{
		AddSpecialTokens:      false, // Keep it simple for tests
		MaxLength:             s.MaxLen,
		Truncation:            true,
		Padding:               true,
		ReturnTokenTypeIds:    true,
		ReturnAttentionMask:   true,
		ReturnOffsetMapping:   false,
		ReturnSpecialTokensMask: false,
		PaddingSide: "right",
		TruncationSide: "right",
	}
}

// Ensure SimpleTestTokenizer satisfies the tokenizer.Tokenizer interface if it's defined.
// For now, this provides the Encode method needed by finetuning.
// We might need to import the actual tokenizer package if an interface is defined.
// For this test, we are assuming finetuning directly uses the Encode method.

// Placeholder for actual tokenizer interface if needed for type compatibility in GradientFineTuner
type TokenizerInterface interface {
    Encode(text string, options *tokenizer.EncodeOptions) (*tokenizer.EncodingResult, error)
    Decode(ids []int, options *tokenizer.DecodeOptions) (string, error)
    GetVocabSize() int
    DefaultEncodeOptions() *tokenizer.EncodeOptions
    // Add other methods if Tokenizer is an interface with more methods used by finetuner
}

var _ TokenizerInterface = (*SimpleTestTokenizer)(nil) // Check compatibility
var _ tokenizer.TokenizerInterface = (*SimpleTestTokenizer)(nil) // Check against actual if it exists


func (s *SimpleTestTokenizer) Normalize(text string) string { return text }
func (s *SimpleTestTokenizer) Tokenize(text string) ([]string, error) {
	return strings.Fields(strings.ToLower(text)), nil
}
func (s *SimpleTestTokenizer) BatchEncode(texts []string, options *tokenizer.BatchEncodeOptions) (*tokenizer.BatchEncodingResult, error) {
	return nil, fmt.Errorf("not implemented for test tokenizer")
}
func (s *SimpleTestTokenizer) BatchDecode(idsList [][]int, options *tokenizer.DecodeOptions) ([]string, error) {
	return nil, fmt.Errorf("not implemented for test tokenizer")
}
func (s *SimpleTestTokenizer) SaveVocabulary(path string) error {
	return fmt.Errorf("not implemented for test tokenizer")
}


```go
package autodiff

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
	"math/rand"


	coreconfig "github.com/transformer_reorganized/pkg/core"
	"github.com/transformer_reorganized/internal/tokenizer"
)

func TestParameterSharingReducesParameterCount(t *testing.T) {
	// Configuration for the models
	config := &coreconfig.Config{
		VocabSize:    1000,
		EmbeddingDim: 64,
		NumLayers:    4, // Using 4 layers to clearly see the effect
		NumHeads:     4,
		FFNHiddenDim: 128,
		MaxLen:       50,
		UseCrossLayerParameterSharing: false,
	}

	// Model 1: No parameter sharing
	model1 := NewTransformerWithTensors(config)
	params1 := model1.GetParameters()
	countParams1 := len(params1)
	t.Logf("Model 1 (No Sharing) Parameter Count: %d", countParams1)

	// Model 2: With parameter sharing
	config.UseCrossLayerParameterSharing = true
	model2 := NewTransformerWithTensors(config)
	params2 := model2.GetParameters()
	countParams2 := len(params2)
	t.Logf("Model 2 (With Sharing) Parameter Count: %d", countParams2)

	// Assertions
	if countParams2 >= countParams1 {
		t.Errorf("Expected parameter count with sharing (%d) to be less than without sharing (%d)", countParams2, countParams1)
	}

	// More specific assertions
	if config.NumLayers > 1 {
		// Count non-embedding/output parameters (layer-specific parameters)
		layerParams1 := 0
		for name := range params1 {
			if !strings.HasPrefix(name, "embedding_") && !strings.HasPrefix(name, "output_") {
				layerParams1++
			}
		}

		layerParams2 := 0
		for name := range params2 {
			if !strings.HasPrefix(name, "shared_") && !strings.HasPrefix(name, "embedding_") && !strings.HasPrefix(name, "output_") {
				// This case should ideally not happen if sharing is effective and GetParameters is correct
				t.Errorf("Found unexpected non-shared, non-embedding/output parameter in shared model: %s", name)
			}
			if strings.HasPrefix(name, "shared_") {
				layerParams2++
			}
		}
		t.Logf("Model 1 Layer-specific params: %d", layerParams1)
		t.Logf("Model 2 Shared Layer-specific params: %d", layerParams2)

		// Expected params per encoder layer (approx): 4 (MHA) + 4 (FFN) + 4 (LN*2) = 12
		// Expected params per decoder layer (approx): 4 (MHA) + 4 (CrossMHA) + 4 (FFN) + 6 (LN*3) = 18
		// Total unique layer params if shared = 12 (for enc) + 18 (for dec) = 30
		// Total unique layer params if not shared = NumLayers * 12 (enc) + NumLayers * 18 (dec)

		// If only encoder layers of NumLayers are present (assuming model structure for simplicity of test)
		// paramsPerEncoderLayer := 12
		// expectedLayerParams1 := config.NumLayers * paramsPerEncoderLayer
		// expectedLayerParams2 := paramsPerEncoderLayer
		// For this test, we check if layerParams2 is much smaller than layerParams1
		if layerParams2 == 0 && layerParams1 > 0 { // Should have some shared params if layers exist
			t.Errorf("Shared model has zero layer-specific parameters, expected some.")
		}
		if layerParams1 == 0 && config.NumLayers > 0 {
			t.Errorf("Non-shared model has zero layer-specific parameters, expected some.")
		}

		if layerParams2 > 0 && layerParams1 > 0 && layerParams1/layerParams2 < config.NumLayers/2 { // Allow some leeway
			t.Errorf("Parameter reduction ratio for layer parameters is less than expected. Shared: %d, Non-shared: %d, NumLayers: %d", layerParams2, layerParams1, config.NumLayers)
		}


		// Check for specific key name patterns
		hasSharedEncoderKey := false
		hasIndexedEncoderKey := false // Should not be present in shared model for layers > 0

		for name := range params2 {
			if strings.HasPrefix(name, "shared_encoder_") { hasSharedEncoderKey = true }
			if strings.HasPrefix(name, "encoder_1_") { hasIndexedEncoderKey = true }
		}

		if config.NumLayers > 0 && !hasSharedEncoderKey {
			// This check assumes there's at least one encoder layer.
			// If the model can have 0 encoder layers but >0 decoder layers and sharing is on, this needs adjustment.
			// For typical transformers, NumLayers applies to both or there's NumEncoderLayers/NumDecoderLayers.
			// Given current Config, NumLayers is generic.
			if len(model2.Encoder) > 0 { // Only check if encoder layers actually exist
				 t.Errorf("Expected 'shared_encoder_*' parameter keys when sharing is enabled, but none found.")
			}
		}
		if hasIndexedEncoderKey {
			t.Errorf("Found indexed encoder parameter key (e.g., 'encoder_1_...') in a shared model.")
		}
	}
}
```

**2. Test Segmented Data Loading (`pkg/autodiff/finetuning_test.go`)**

I'll create this file and the mock tokenizer within it for simplicity.
