package autodiff

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix represents a 2D matrix of float64 values
type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

// NewMatrix creates a new matrix with the specified dimensions
func NewMatrix(rows, cols int) (*Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid matrix dimensions: rows=%d, cols=%d (must be positive)", rows, cols)
	}

	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}

	return &Matrix{
		Rows: rows,
		Cols: cols,
		Data: data,
	}, nil
}

// MustNewMatrix creates a new matrix with the specified dimensions
// Panics if dimensions are invalid (use in non-production code only)
func MustNewMatrix(rows, cols int) *Matrix {
	m, err := NewMatrix(rows, cols)
	if err != nil {
		panic(err)
	}
	return m
}

// NewRandomMatrix creates a new matrix with random values
func NewRandomMatrix(rows, cols int) (*Matrix, error) {
	m, err := NewMatrix(rows, cols)
	if err != nil {
		return nil, err
	}

	// Initialize with small random values for better training stability
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float64()*0.2 - 0.1
		}
	}

	return m, nil
}

// MustNewRandomMatrix creates a new matrix with random values
// Panics if dimensions are invalid (use in non-production code only)
func MustNewRandomMatrix(rows, cols int) *Matrix {
	m, err := NewRandomMatrix(rows, cols)
	if err != nil {
		panic(err)
	}
	return m
}

// Clone creates a deep copy of the matrix
func (m *Matrix) Clone() (*Matrix, error) {
	clone, err := NewMatrix(m.Rows, m.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			clone.Data[i][j] = m.Data[i][j]
		}
	}

	return clone, nil
}

// MustClone creates a deep copy of the matrix
// Panics if an error occurs (use in non-production code only)
func (m *Matrix) MustClone() *Matrix {
	clone, err := m.Clone()
	if err != nil {
		panic(err)
	}
	return clone
}

// MatMul performs matrix multiplication
func MatMul(a, b *Matrix) (*Matrix, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("cannot multiply nil matrices")
	}

	if a.Cols != b.Rows {
		return nil, fmt.Errorf("matrix dimensions don't match for multiplication: a(%dx%d), b(%dx%d)",
			a.Rows, a.Cols, b.Rows, b.Cols)
	}

	result, err := NewMatrix(a.Rows, b.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < b.Cols; j++ {
			sum := 0.0
			for k := 0; k < a.Cols; k++ {
				sum += a.Data[i][k] * b.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}

	return result, nil
}

// MustMatMul performs matrix multiplication
// Panics if an error occurs (use in non-production code only)
func MustMatMul(a, b *Matrix) *Matrix {
	result, err := MatMul(a, b)
	if err != nil {
		panic(err)
	}
	return result
}

// Add adds two matrices element-wise
func Add(a, b *Matrix) (*Matrix, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("cannot add nil matrices")
	}

	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("matrix dimensions don't match for addition: a(%dx%d), b(%dx%d)",
			a.Rows, a.Cols, b.Rows, b.Cols)
	}

	result, err := NewMatrix(a.Rows, a.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] + b.Data[i][j]
		}
	}

	return result, nil
}

// MustAdd adds two matrices element-wise
// Panics if an error occurs (use in non-production code only)
func MustAdd(a, b *Matrix) *Matrix {
	result, err := Add(a, b)
	if err != nil {
		panic(err)
	}
	return result
}

// Subtract subtracts matrix b from matrix a element-wise
func Subtract(a, b *Matrix) (*Matrix, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("cannot subtract nil matrices")
	}

	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("matrix dimensions don't match for subtraction: a(%dx%d), b(%dx%d)",
			a.Rows, a.Cols, b.Rows, b.Cols)
	}

	result, err := NewMatrix(a.Rows, a.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] - b.Data[i][j]
		}
	}

	return result, nil
}

// MustSubtract subtracts matrix b from matrix a element-wise
// Panics if an error occurs (use in non-production code only)
func MustSubtract(a, b *Matrix) *Matrix {
	result, err := Subtract(a, b)
	if err != nil {
		panic(err)
	}
	return result
}

// Multiply performs element-wise multiplication (Hadamard product)
func Multiply(a, b *Matrix) (*Matrix, error) {
	if a == nil || b == nil {
		return nil, fmt.Errorf("cannot multiply nil matrices")
	}

	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, fmt.Errorf("matrix dimensions don't match for element-wise multiplication: a(%dx%d), b(%dx%d)",
			a.Rows, a.Cols, b.Rows, b.Cols)
	}

	result, err := NewMatrix(a.Rows, a.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			result.Data[i][j] = a.Data[i][j] * b.Data[i][j]
		}
	}

	return result, nil
}

// MustMultiply performs element-wise multiplication (Hadamard product)
// Panics if an error occurs (use in non-production code only)
func MustMultiply(a, b *Matrix) *Matrix {
	result, err := Multiply(a, b)
	if err != nil {
		panic(err)
	}
	return result
}

// ScalarMultiply multiplies all elements of a matrix by a scalar value
func ScalarMultiply(m *Matrix, scalar float64) (*Matrix, error) {
	if m == nil {
		return nil, fmt.Errorf("cannot multiply nil matrix by scalar")
	}

	result, err := NewMatrix(m.Rows, m.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}

	return result, nil
}

// MustScalarMultiply multiplies all elements of a matrix by a scalar value
// Panics if an error occurs (use in non-production code only)
func MustScalarMultiply(m *Matrix, scalar float64) *Matrix {
	result, err := ScalarMultiply(m, scalar)
	if err != nil {
		panic(err)
	}
	return result
}

// Transpose returns the transpose of a matrix
func Transpose(m *Matrix) (*Matrix, error) {
	if m == nil {
		return nil, fmt.Errorf("cannot transpose nil matrix")
	}

	result, err := NewMatrix(m.Cols, m.Rows)
	if err != nil {
		return nil, err
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}

	return result, nil
}

// MustTranspose returns the transpose of a matrix
// Panics if an error occurs (use in non-production code only)
func MustTranspose(m *Matrix) *Matrix {
	result, err := Transpose(m)
	if err != nil {
		panic(err)
	}
	return result
}

// Softmax applies the softmax function to each row of the matrix
func Softmax(m *Matrix) (*Matrix, error) {
	if m == nil {
		return nil, fmt.Errorf("cannot apply softmax to nil matrix")
	}

	result, err := NewMatrix(m.Rows, m.Cols)
	if err != nil {
		return nil, err
	}

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

	return result, nil
}

// MustSoftmax applies the softmax function to each row of the matrix
// Panics if an error occurs (use in non-production code only)
func MustSoftmax(m *Matrix) *Matrix {
	result, err := Softmax(m)
	if err != nil {
		panic(err)
	}
	return result
}

// ApplyFunction applies a function to each element of the matrix
func ApplyFunction(m *Matrix, fn func(float64) float64) (*Matrix, error) {
	if m == nil {
		return nil, fmt.Errorf("cannot apply function to nil matrix")
	}

	result, err := NewMatrix(m.Rows, m.Cols)
	if err != nil {
		return nil, err
	}

	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}

	return result, nil
}

// MustApplyFunction applies a function to each element of the matrix
// Panics if an error occurs (use in non-production code only)
func MustApplyFunction(m *Matrix, fn func(float64) float64) *Matrix {
	result, err := ApplyFunction(m, fn)
	if err != nil {
		panic(err)
	}
	return result
}

// Sum returns the sum of all elements in the matrix
func Sum(m *Matrix) (float64, error) {
	if m == nil {
		return 0, fmt.Errorf("cannot sum nil matrix")
	}

	sum := 0.0
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sum += m.Data[i][j]
		}
	}

	return sum, nil
}

// MustSum returns the sum of all elements in the matrix
// Panics if an error occurs (use in non-production code only)
func MustSum(m *Matrix) float64 {
	sum, err := Sum(m)
	if err != nil {
		panic(err)
	}
	return sum
}

// Mean returns the mean of all elements in the matrix
func Mean(m *Matrix) (float64, error) {
	if m == nil {
		return 0, fmt.Errorf("cannot calculate mean of nil matrix")
	}

	if m.Rows == 0 || m.Cols == 0 {
		return 0, fmt.Errorf("cannot calculate mean of empty matrix")
	}

	sum, err := Sum(m)
	if err != nil {
		return 0, err
	}

	return sum / float64(m.Rows*m.Cols), nil
}

// MustMean returns the mean of all elements in the matrix
// Panics if an error occurs (use in non-production code only)
func MustMean(m *Matrix) float64 {
	mean, err := Mean(m)
	if err != nil {
		panic(err)
	}
	return mean
}

// Equal checks if two matrices are equal (same dimensions and values)
func Equal(a, b *Matrix, epsilon float64) (bool, error) {
	if a == nil || b == nil {
		return false, fmt.Errorf("cannot compare nil matrices")
	}

	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false, nil
	}

	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			if math.Abs(a.Data[i][j] - b.Data[i][j]) > epsilon {
				return false, nil
			}
		}
	}

	return true, nil
}

// MustEqual checks if two matrices are equal (same dimensions and values)
// Panics if an error occurs (use in non-production code only)
func MustEqual(a, b *Matrix, epsilon float64) bool {
	equal, err := Equal(a, b, epsilon)
	if err != nil {
		panic(err)
	}
	return equal
}

// String returns a string representation of the matrix
func (m *Matrix) String() string {
	if m == nil {
		return "nil"
	}

	result := fmt.Sprintf("Matrix(%dx%d):\n", m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		result += "["
		for j := 0; j < m.Cols; j++ {
			if j > 0 {
				result += ", "
			}
			result += fmt.Sprintf("%.4f", m.Data[i][j])
		}
		result += "]\n"
	}

	return result
}

// Legacy compatibility functions to support older code
// These should be deprecated in future versions

// LegacyNewMatrix creates a new matrix without error checking
func LegacyNewMatrix(rows, cols int) *Matrix {
	return MustNewMatrix(rows, cols)
}

// LegacyNewRandomMatrix creates a new random matrix without error checking
func LegacyNewRandomMatrix(rows, cols int) *Matrix {
	return MustNewRandomMatrix(rows, cols)
}

// LegacyMatMul performs matrix multiplication without error checking
func LegacyMatMul(a, b *Matrix) *Matrix {
	return MustMatMul(a, b)
}

// LegacySoftmax applies softmax without error checking
func LegacySoftmax(m *Matrix) *Matrix {
	return MustSoftmax(m)
}
