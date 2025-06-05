package core

import (
	// "math" // Removed unused import
	"fmt" // Added fmt import
	"transformer/internal/utils" // Added utils import
)

// EnhancedPositionalEncoding represents an advanced positional encoding with more options
type EnhancedPositionalEncoding struct {
	Dim       int
	MaxLen    int
	Encoding  *utils.Matrix // Prefixed with utils.
	DropPE    *utils.Dropout // Prefixed with utils.
	Scale     float64
	Learnable bool
}

// NewEnhancedPositionalEncoding is defined in merged_positional.go
/*
func NewEnhancedPositionalEncoding(dim, maxLen int, scale float64, dropoutRate float64, learnable bool) *EnhancedPositionalEncoding {
// 	encoding := utils.NewMatrix(maxLen, dim) // Prefixed with utils.
	
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
*/

// AddToEmbedding adds positional encoding to the input embeddings
func (pe *EnhancedPositionalEncoding) AddToEmbedding(embeddings *utils.Matrix, isTraining bool) (*utils.Matrix, error) { // Added error return, prefixed Matrix
	if embeddings.Rows > pe.MaxLen {
		// Consider returning an error instead of panicking
		return nil, fmt.Errorf("input sequence length (%d) exceeds maximum length (%d) for positional encoding", embeddings.Rows, pe.MaxLen)
	}
	
	result, err := utils.NewMatrix(embeddings.Rows, embeddings.Cols) // Prefixed with utils.
	if err != nil {
		return nil, fmt.Errorf("failed to create result matrix: %w", err)
	}
	
	posEnc, err := utils.NewMatrix(embeddings.Rows, embeddings.Cols) // Prefixed with utils.
	if err != nil {
		return nil, fmt.Errorf("failed to create posEnc matrix: %w", err)
	}

	for i := 0; i < embeddings.Rows; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			posEnc.Data[i][j] = pe.Encoding.Data[i][j]
		}
	}
	
	if isTraining && pe.DropPE.Rate > 0 {
		posEncWithDropout, errDropout := pe.DropPE.Forward(posEnc, isTraining)
		if errDropout != nil {
			return nil, fmt.Errorf("dropout on posEnc failed: %w", errDropout)
		}
		posEnc = posEncWithDropout
	}
	
	if pe.Scale != 1.0 {
		for i := 0; i < posEnc.Rows; i++ {
			for j := 0; j < posEnc.Cols; j++ {
				posEnc.Data[i][j] *= pe.Scale
			}
		}
	}
	
	for i := 0; i < embeddings.Rows; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			result.Data[i][j] = embeddings.Data[i][j] + posEnc.Data[i][j]
		}
	}
	
	return result, nil
}

// RotaryPositionalEncoding implements Rotary Position Embedding (RoPE)
type RotaryPositionalEncoding struct {
	Dim    int
	MaxLen int
	Cos    *utils.Matrix // Prefixed with utils.
	Sin    *utils.Matrix // Prefixed with utils.
}

// NewRotaryPositionalEncoding is defined in merged_positional.go
/*
func NewRotaryPositionalEncoding(dim, maxLen int) *RotaryPositionalEncoding {
	cos := utils.NewMatrix(maxLen, dim/2) // Prefixed with utils.
	sin := utils.NewMatrix(maxLen, dim/2) // Prefixed with utils.
	
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
*/

// ApplyRotary applies rotary position embeddings to query and key tensors
func (rpe *RotaryPositionalEncoding) ApplyRotary(x *utils.Matrix, seqLen int) (*utils.Matrix, error) { // Added error return, prefixed Matrix
	if seqLen > rpe.MaxLen {
		// Consider returning an error
		return nil, fmt.Errorf("sequence length (%d) exceeds maximum length (%d) for rotary encoding", seqLen, rpe.MaxLen)
	}
	
	result, err := utils.NewMatrix(x.Rows, x.Cols) // Prefixed with utils.
	if err != nil {
		return nil, fmt.Errorf("failed to create result matrix for ApplyRotary: %w", err)
	}
	
	halfDim := x.Cols / 2
	
	for i := 0; i < x.Rows; i++ {
		pos := i % seqLen
		
		for j := 0; j < halfDim; j++ {
			cosVal := rpe.Cos.Data[pos][j]
			sinVal := rpe.Sin.Data[pos][j]
			
			x1 := x.Data[i][j]
			x2 := x.Data[i][j+halfDim]
			
			result.Data[i][j] = x1*cosVal - x2*sinVal
			result.Data[i][j+halfDim] = x1*sinVal + x2*cosVal
		}
	}
	
	return result, nil
}
