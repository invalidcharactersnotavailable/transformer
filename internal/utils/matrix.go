package utils

// Matrix represents a 2D matrix of float64 values
type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

// NewMatrix creates a new matrix with specified dimensions
func NewMatrix(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}
}

// NewRandomMatrix creates a new matrix with random values
func NewRandomMatrix(rows, cols int) *Matrix {
	matrix := NewMatrix(rows, cols)
	// Initialize with random values (simplified for validation)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix.Data[i][j] = float64(i+j) / float64(rows+cols) // Simplified random value
		}
	}
	return matrix
}

// MatMul performs matrix multiplication
func MatMul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic("Matrix dimensions don't match for multiplication")
	}
	
	result := NewMatrix(a.Rows, b.Cols)
	
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	
	return result
}

// Softmax applies the softmax function to the matrix
func Softmax(m *Matrix) *Matrix {
	result := NewMatrix(m.Rows, m.Cols)
	
	for i := 0; i < m.Rows; i++ {
		// Find max value in row for numerical stability
		max := m.Data[i][0]
		for j := 1; j < m.Cols; j++ {
			if m.Data[i][j] > max {
				max = m.Data[i][j]
			}
		}
		
		// Calculate sum of exponentials
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			sum += exp(m.Data[i][j] - max)
		}
		
		// Apply softmax
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = exp(m.Data[i][j] - max) / sum
		}
	}
	
	return result
}

// Simple exponential function for validation purposes
func exp(x float64) float64 {
	// Simplified exponential approximation for validation
	if x > 20 {
		return 1e9
	} else if x < -20 {
		return 1e-9
	}
	
	result := 1.0
	term := 1.0
	for i := 1; i < 10; i++ {
		term *= x / float64(i)
		result += term
	}
	return result
}
