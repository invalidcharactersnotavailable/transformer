package core

import (
	"fmt"
	"math"
	"transformer/internal/utils" // Added utils import
)

// PositionalEncodingType represents the type of positional encoding to use
type PositionalEncodingType int

const (
	// SinusoidalEncoding uses fixed sinusoidal patterns
	SinusoidalEncoding PositionalEncodingType = iota
	// RotaryEncoding uses Rotary Position Embedding (RoPE)
	RotaryEncoding
	// LearnableEncoding uses learnable position embeddings
	LearnableEncoding
)

// PositionalEncodingConfig holds configuration options for positional encoding
type PositionalEncodingConfig struct {
	Dim           int
	MaxLen        int
	DropoutRate   float64
	Scale         float64
	EncodingType  PositionalEncodingType
	Learnable     bool
}

// DefaultPositionalEncodingConfig returns the default configuration for positional encoding
func DefaultPositionalEncodingConfig() *PositionalEncodingConfig {
	return &PositionalEncodingConfig{
		Dim:          512,
		MaxLen:       1024,
		DropoutRate:  0.1,
		Scale:        1.0,
		EncodingType: SinusoidalEncoding,
		Learnable:    false,
	}
}

// PositionalEncoding represents a unified positional encoding interface
type PositionalEncoding struct {
	Dim           int
	MaxLen        int
	Encoding      *utils.Matrix // Prefixed
	DropPE        *utils.Dropout // Prefixed
	Scale         float64
	EncodingType  PositionalEncodingType
	Learnable     bool
	
	// For rotary encoding
	Cos           *utils.Matrix // Prefixed
	Sin           *utils.Matrix // Prefixed
	
	// For learnable encoding
	LearnableWeights *utils.Matrix // Prefixed
}

// NewPositionalEncoding creates a new positional encoding with the specified configuration
func NewPositionalEncoding(config *PositionalEncodingConfig) (*PositionalEncoding, error) {
	if config == nil {
		config = DefaultPositionalEncodingConfig()
	}
	
	if config.Dim <= 0 {
		return nil, fmt.Errorf("dimension must be positive, got %d", config.Dim)
	}
	
	if config.MaxLen <= 0 {
		return nil, fmt.Errorf("maximum length must be positive, got %d", config.MaxLen)
	}
	
	if config.DropoutRate < 0 || config.DropoutRate >= 1.0 {
		return nil, fmt.Errorf("dropout rate must be in range [0, 1), got %f", config.DropoutRate)
	}
	
	// Note: utils.NewDropout now doesn't return error, if it did, this would need handling
	dropPE := utils.NewDropout(config.DropoutRate)

	pe := &PositionalEncoding{
		Dim:          config.Dim,
		MaxLen:       config.MaxLen,
		DropPE:       dropPE, // Use the initialized one
		Scale:        config.Scale,
		EncodingType: config.EncodingType,
		Learnable:    config.Learnable,
	}
	
	// Initialize based on encoding type
	switch config.EncodingType {
	case SinusoidalEncoding:
		encoding, err := utils.NewMatrix(config.MaxLen, config.Dim) // Prefixed
		if err != nil {
			return nil, fmt.Errorf("failed to create encoding matrix: %v", err)
		}
		
		// Initialize with sinusoidal encoding
		for pos := 0; pos < config.MaxLen; pos++ {
			for i := 0; i < config.Dim; i += 2 {
				denominator := math.Pow(10000, float64(i)/float64(config.Dim))
				
				// Sine for even indices
				if i < config.Dim {
					encoding.Data[pos][i] = math.Sin(float64(pos) / denominator)
				}
				
				// Cosine for odd indices
				if i+1 < config.Dim {
					encoding.Data[pos][i+1] = math.Cos(float64(pos) / denominator)
				}
			}
		}
		
		pe.Encoding = encoding
		
	case RotaryEncoding:
		if config.Dim%2 != 0 {
			return nil, fmt.Errorf("dimension must be even for rotary encoding, got %d", config.Dim)
		}
		
		cos, err := utils.NewMatrix(config.MaxLen, config.Dim/2) // Prefixed
		if err != nil {
			return nil, fmt.Errorf("failed to create cos matrix: %v", err)
		}
		
		sin, err := utils.NewMatrix(config.MaxLen, config.Dim/2) // Prefixed
		if err != nil {
			return nil, fmt.Errorf("failed to create sin matrix: %v", err)
		}
		
		for pos := 0; pos < config.MaxLen; pos++ {
			for i := 0; i < config.Dim/2; i++ {
				freq := 1.0 / math.Pow(10000, float64(2*i)/float64(config.Dim))
				cos.Data[pos][i] = math.Cos(float64(pos) * freq)
				sin.Data[pos][i] = math.Sin(float64(pos) * freq)
			}
		}
		
		pe.Cos = cos
		pe.Sin = sin
		
	case LearnableEncoding:
		// Initialize learnable weights with small random values
		weights, err := utils.NewRandomMatrix(config.MaxLen, config.Dim) // Prefixed
		if err != nil {
			return nil, fmt.Errorf("failed to create learnable weights: %v", err)
		}
		
		// Scale down initial values for better training stability
		for i := 0; i < weights.Rows; i++ {
			for j := 0; j < weights.Cols; j++ {
				weights.Data[i][j] *= 0.02
			}
		}
		
		pe.LearnableWeights = weights
		
	default:
		return nil, fmt.Errorf("unknown encoding type: %v", config.EncodingType)
	}
	
	return pe, nil
}

// NewStandardPositionalEncoding creates a new standard sinusoidal positional encoding
func NewStandardPositionalEncoding(dim, maxLen int) (*PositionalEncoding, error) {
	return NewPositionalEncoding(&PositionalEncodingConfig{
		Dim:          dim,
		MaxLen:       maxLen,
		DropoutRate:  0.0,
		Scale:        1.0,
		EncodingType: SinusoidalEncoding,
		Learnable:    false,
	})
}

// NewEnhancedPositionalEncoding creates a new enhanced positional encoding with dropout and scaling
func NewEnhancedPositionalEncoding(dim, maxLen int, scale float64, dropoutRate float64, learnable bool) (*PositionalEncoding, error) {
	encodingType := SinusoidalEncoding
	if learnable {
		encodingType = LearnableEncoding
	}
	
	return NewPositionalEncoding(&PositionalEncodingConfig{
		Dim:          dim,
		MaxLen:       maxLen,
		DropoutRate:  dropoutRate,
		Scale:        scale,
		EncodingType: encodingType,
		Learnable:    learnable,
	})
}

// NewRotaryPositionalEncoding creates a new rotary positional encoding
func NewRotaryPositionalEncoding(dim, maxLen int) (*PositionalEncoding, error) {
	return NewPositionalEncoding(&PositionalEncodingConfig{
		Dim:          dim,
		MaxLen:       maxLen,
		DropoutRate:  0.0,
		Scale:        1.0,
		EncodingType: RotaryEncoding,
		Learnable:    false,
	})
}

// AddToEmbedding adds positional encoding to the input embeddings
func (pe *PositionalEncoding) AddToEmbedding(embeddings *utils.Matrix, isTraining bool) (*utils.Matrix, error) { // Prefixed
	if embeddings == nil {
		return nil, fmt.Errorf("embeddings cannot be nil")
	}
	
	if embeddings.Rows > pe.MaxLen {
		return nil, fmt.Errorf("input sequence length (%d) exceeds maximum length (%d) for positional encoding", 
			embeddings.Rows, pe.MaxLen)
	}
	
	if embeddings.Cols != pe.Dim && pe.EncodingType != RotaryEncoding {
		return nil, fmt.Errorf("embedding dimension (%d) doesn't match positional encoding dimension (%d)", 
			embeddings.Cols, pe.Dim)
	}
	
	// For rotary encoding, we don't add but apply rotation
	if pe.EncodingType == RotaryEncoding {
		return pe.ApplyRotary(embeddings, embeddings.Rows)
	}
	
	result, err := utils.NewMatrix(embeddings.Rows, embeddings.Cols) // Prefixed
	if err != nil {
		return nil, fmt.Errorf("failed to create result matrix: %v", err)
	}
	
	// Extract positional encodings for the current sequence length
	var posEnc *utils.Matrix // Prefixed
	
	if pe.EncodingType == LearnableEncoding {
		// Use learnable weights
		// var err error // err already declared by result
		posEnc, err = utils.NewMatrix(embeddings.Rows, embeddings.Cols) // Prefixed
		if err != nil {
			return nil, fmt.Errorf("failed to create position encoding matrix: %v", err)
		}
		
		for i := 0; i < embeddings.Rows; i++ {
			for j := 0; j < embeddings.Cols; j++ {
				posEnc.Data[i][j] = pe.LearnableWeights.Data[i][j]
			}
		}
	} else {
		// Use fixed sinusoidal encoding
		// var err error // err already declared by result
		posEnc, err = utils.NewMatrix(embeddings.Rows, embeddings.Cols) // Prefixed
		if err != nil {
			return nil, fmt.Errorf("failed to create position encoding matrix: %v", err)
		}
		
		for i := 0; i < embeddings.Rows; i++ {
			for j := 0; j < embeddings.Cols; j++ {
				posEnc.Data[i][j] = pe.Encoding.Data[i][j]
			}
		}
	}
	
	// Apply dropout to positional encoding if training
	if isTraining && pe.DropPE.Rate > 0 {
		posEncWithDropout, errDropout := pe.DropPE.Forward(posEnc, isTraining) // utils.Dropout.Forward now returns error
		if errDropout != nil {
			return nil, fmt.Errorf("dropout on posEnc failed: %w", errDropout)
		}
		posEnc = posEncWithDropout
	}
	
	// Scale positional encoding if needed
	if pe.Scale != 1.0 {
		for i := 0; i < posEnc.Rows; i++ {
			for j := 0; j < posEnc.Cols; j++ {
				posEnc.Data[i][j] *= pe.Scale
			}
		}
	}
	
	// Add to embeddings
	for i := 0; i < embeddings.Rows; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			result.Data[i][j] = embeddings.Data[i][j] + posEnc.Data[i][j]
		}
	}
	
	return result, nil
}

// ApplyRotary applies rotary position embeddings to input tensors
func (pe *PositionalEncoding) ApplyRotary(x *utils.Matrix, seqLen int) (*utils.Matrix, error) { // Prefixed
	if x == nil {
		return nil, fmt.Errorf("input matrix cannot be nil")
	}
	
	if pe.EncodingType != RotaryEncoding {
		return nil, fmt.Errorf("rotary application only supported for rotary encoding type")
	}
	
	if seqLen > pe.MaxLen {
		return nil, fmt.Errorf("sequence length (%d) exceeds maximum length (%d) for rotary encoding", 
			seqLen, pe.MaxLen)
	}
	
	if x.Cols%2 != 0 {
		return nil, fmt.Errorf("input dimension must be even for rotary encoding, got %d", x.Cols)
	}
	
	result, err := utils.NewMatrix(x.Rows, x.Cols) // Prefixed
	if err != nil {
		return nil, fmt.Errorf("failed to create result matrix: %v", err)
	}
	
	// Apply rotary position embeddings
	halfDim := x.Cols / 2
	
	for i := 0; i < x.Rows; i++ {
		pos := i % seqLen
		
		for j := 0; j < halfDim; j++ {
			// Apply rotation to each pair of dimensions
			cos := pe.Cos.Data[pos][j]
			sin := pe.Sin.Data[pos][j]
			
			// First half of the dimension
			x1 := x.Data[i][j]
			x2 := x.Data[i][j+halfDim]
			
			// Rotate using complex multiplication
			result.Data[i][j] = x1*cos - x2*sin
			result.Data[i][j+halfDim] = x1*sin + x2*cos
		}
	}
	
	return result, nil
}

// AddToEmbeddingLegacy provides backward compatibility with the original API
func (pe *PositionalEncoding) AddToEmbeddingLegacy(embeddings *utils.Matrix) *utils.Matrix { // Prefixed
	result, err := pe.AddToEmbedding(embeddings, false)
	if err != nil {
		// In legacy mode, return a zero matrix on error
		zeroMatrix, _ := utils.NewMatrix(embeddings.Rows, embeddings.Cols) // Prefixed
		return zeroMatrix
	}
	return result
}

// Clone creates a deep copy of the PositionalEncoding
func (pe *PositionalEncoding) Clone() (*PositionalEncoding, error) {
	clone := &PositionalEncoding{
		Dim:          pe.Dim,
		MaxLen:       pe.MaxLen,
		DropPE:       pe.DropPE.Clone(),
		Scale:        pe.Scale,
		EncodingType: pe.EncodingType,
		Learnable:    pe.Learnable,
	}
	
	var err error
	
	// Clone appropriate fields based on encoding type
	switch pe.EncodingType {
	case SinusoidalEncoding:
		if pe.Encoding != nil {
			clone.Encoding, err = pe.Encoding.Clone()
			if err != nil {
				return nil, fmt.Errorf("failed to clone encoding matrix: %v", err)
			}
		}
		
	case RotaryEncoding:
		if pe.Cos != nil {
			clone.Cos, err = pe.Cos.Clone()
			if err != nil {
				return nil, fmt.Errorf("failed to clone cos matrix: %v", err)
			}
		}
		
		if pe.Sin != nil {
			clone.Sin, err = pe.Sin.Clone()
			if err != nil {
				return nil, fmt.Errorf("failed to clone sin matrix: %v", err)
			}
		}
		
	case LearnableEncoding:
		if pe.LearnableWeights != nil {
			clone.LearnableWeights, err = pe.LearnableWeights.Clone()
			if err != nil {
				return nil, fmt.Errorf("failed to clone learnable weights: %v", err)
			}
		}
	}
	
	return clone, nil
}

// Legacy compatibility functions to support older code
// These should be deprecated in future versions

// LegacyNewPositionalEncoding creates a new standard positional encoding without error checking
func LegacyNewPositionalEncoding(dim, maxLen int) *PositionalEncoding {
	pe, _ := NewStandardPositionalEncoding(dim, maxLen)
	return pe
}

// LegacyNewEnhancedPositionalEncoding creates a new enhanced positional encoding without error checking
func LegacyNewEnhancedPositionalEncoding(dim, maxLen int, scale float64, dropoutRate float64, learnable bool) *PositionalEncoding {
	pe, _ := NewEnhancedPositionalEncoding(dim, maxLen, scale, dropoutRate, learnable)
	return pe
}

// LegacyNewRotaryPositionalEncoding creates a new rotary positional encoding without error checking
func LegacyNewRotaryPositionalEncoding(dim, maxLen int) *PositionalEncoding {
	pe, _ := NewRotaryPositionalEncoding(dim, maxLen)
	return pe
}
