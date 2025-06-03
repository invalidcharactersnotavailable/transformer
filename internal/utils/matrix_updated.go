package transformer

import (
	"math"
)

// Matrix represents a 2D matrix of float64 values
type Matrix struct {
	Rows    int
	Cols    int
	Data    [][]float64
}

// NewMatrix creates a new matrix with the specified dimensions
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
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float64()*0.2 - 0.1 // Initialize with small random values
		}
	}
	return m
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

// Add adds two matrices element-wise
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("Matrix dimensions don't match for addition")
	}
	
	result := NewMatrix(a.Rows, a.Cols)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] + b.Data[i][j]
		}
	}
	return result
}

// Softmax applies the softmax function to each row of the matrix
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
		
		// Calculate exp and sum
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Exp(m.Data[i][j] - max)
			sum += result.Data[i][j]
		}
		
		// Normalize
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] /= sum
		}
	}
	
	return result
}

// Transpose returns the transpose of a matrix
func Transpose(m *Matrix) *Matrix {
	result := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}
