package transformer

import (
	"math"
)

// EnhancedPositionalEncoding represents an advanced positional encoding with more options
type EnhancedPositionalEncoding struct {
	Dim       int
	MaxLen    int
	Encoding  *Matrix
	DropPE    *Dropout
	Scale     float64
	Learnable bool
}

// NewEnhancedPositionalEncoding creates a new enhanced positional encoding
func NewEnhancedPositionalEncoding(dim, maxLen int, scale float64, dropoutRate float64, learnable bool) *EnhancedPositionalEncoding {
	encoding := NewMatrix(maxLen, dim)
	
	// Initialize with sinusoidal encoding
	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dim; i += 2 {
			denominator := math.Pow(10000, float64(i)/float64(dim))
			
			// Sine for even indices
			if i < dim {
				encoding.Data[pos][i] = math.Sin(float64(pos) / denominator)
			}
			
			// Cosine for odd indices
			if i+1 < dim {
				encoding.Data[pos][i+1] = math.Cos(float64(pos) / denominator)
			}
		}
	}
	
	return &EnhancedPositionalEncoding{
		Dim:       dim,
		MaxLen:    maxLen,
		Encoding:  encoding,
		DropPE:    NewDropout(dropoutRate),
		Scale:     scale,
		Learnable: learnable,
	}
}

// AddToEmbedding adds positional encoding to the input embeddings
func (pe *EnhancedPositionalEncoding) AddToEmbedding(embeddings *Matrix, isTraining bool) *Matrix {
	if embeddings.Rows > pe.MaxLen {
		panic("Input sequence length exceeds maximum length for positional encoding")
	}
	
	result := NewMatrix(embeddings.Rows, embeddings.Cols)
	
	// Extract positional encodings for the current sequence length
	posEnc := NewMatrix(embeddings.Rows, embeddings.Cols)
	for i := 0; i < embeddings.Rows; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			posEnc.Data[i][j] = pe.Encoding.Data[i][j]
		}
	}
	
	// Apply dropout to positional encoding if training
	if isTraining && pe.DropPE.Rate > 0 {
		posEnc = pe.DropPE.Forward(posEnc, isTraining)
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
	
	return result
}

// RotaryPositionalEncoding implements Rotary Position Embedding (RoPE)
type RotaryPositionalEncoding struct {
	Dim    int
	MaxLen int
	Cos    *Matrix
	Sin    *Matrix
}

// NewRotaryPositionalEncoding creates a new rotary positional encoding
func NewRotaryPositionalEncoding(dim, maxLen int) *RotaryPositionalEncoding {
	cos := NewMatrix(maxLen, dim/2)
	sin := NewMatrix(maxLen, dim/2)
	
	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dim/2; i++ {
			freq := 1.0 / math.Pow(10000, float64(2*i)/float64(dim))
			cos.Data[pos][i] = math.Cos(float64(pos) * freq)
			sin.Data[pos][i] = math.Sin(float64(pos) * freq)
		}
	}
	
	return &RotaryPositionalEncoding{
		Dim:    dim,
		MaxLen: maxLen,
		Cos:    cos,
		Sin:    sin,
	}
}

// ApplyRotary applies rotary position embeddings to query and key tensors
func (rpe *RotaryPositionalEncoding) ApplyRotary(x *Matrix, seqLen int) *Matrix {
	if seqLen > rpe.MaxLen {
		panic("Sequence length exceeds maximum length for rotary encoding")
	}
	
	result := NewMatrix(x.Rows, x.Cols)
	
	// This is a simplified implementation of RoPE
	// In a real implementation, we would apply complex rotation to pairs of dimensions
	halfDim := x.Cols / 2
	
	for i := 0; i < x.Rows; i++ {
		pos := i % seqLen
		
		for j := 0; j < halfDim; j++ {
			// Apply rotation to each pair of dimensions
			cos := rpe.Cos.Data[pos][j]
			sin := rpe.Sin.Data[pos][j]
			
			// First half of the dimension
			x1 := x.Data[i][j]
			x2 := x.Data[i][j+halfDim]
			
			// Rotate using complex multiplication
			result.Data[i][j] = x1*cos - x2*sin
			result.Data[i][j+halfDim] = x1*sin + x2*cos
		}
	}
	
	return result
}
