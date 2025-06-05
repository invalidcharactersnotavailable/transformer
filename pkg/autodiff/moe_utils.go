package autodiff

import "fmt"

// NewMatrixZeros creates a new matrix initialized with all zeros.
// This is used by MoELayer.Forward for bypassed_aux_loss and topk_err_aux_loss.
// It assumes NewMatrix initializes to zeros, which it does.
func NewMatrixZeros(rows, cols int) (*Matrix, error) {
	return NewMatrix(rows, cols)
}

// NewMatrixFromSlice creates a matrix from a flat slice of data.
// Used by MoELayer.Forward for f_i_tensor.
func NewMatrixFromSlice(data []float64, rows, cols int) (*Matrix, error) {
	if rows*cols != len(data) {
		return nil, fmt.Errorf("dimension mismatch: %d*%d != %d", rows, cols, len(data))
	}
	matrix, err := NewMatrix(rows, cols)
	if err != nil {
		return nil, err
	}
	idx := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			matrix.Data[i][j] = data[idx]
			idx++
		}
	}
	return matrix, nil
}
