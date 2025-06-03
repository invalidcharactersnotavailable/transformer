package transformer

// Batch represents a batch of sequences for efficient processing
type Batch struct {
	Sequences      *Matrix
	AttentionMask  *AttentionMask
	SequenceLengths []int
	BatchSize      int
}

// NewBatch creates a new batch from sequences
func NewBatch(sequences [][]int, maxLen int) *Batch {
	batchSize := len(sequences)
	
	// Create matrix to hold sequences
	batchMatrix := NewMatrix(batchSize, maxLen)
	
	// Track actual sequence lengths
	sequenceLengths := make([]int, batchSize)
	
	// Fill matrix with sequence IDs, padding with zeros
	for i, seq := range sequences {
		sequenceLengths[i] = len(seq)
		for j := 0; j < len(seq) && j < maxLen; j++ {
			batchMatrix.Data[i][j] = float64(seq[j])
		}
	}
	
	// Create padding mask
	paddingMask := NewPaddingMask(maxLen, sequenceLengths)
	
	return &Batch{
		Sequences:      batchMatrix,
		AttentionMask:  paddingMask,
		SequenceLengths: sequenceLengths,
		BatchSize:      batchSize,
	}
}

// GetEmbeddings converts a batch of token IDs to embeddings
func (b *Batch) GetEmbeddings(embeddings *Matrix) *Matrix {
	result := NewMatrix(b.BatchSize, embeddings.Cols)
	
	for i := 0; i < b.BatchSize; i++ {
		for j := 0; j < b.SequenceLengths[i]; j++ {
			tokenID := int(b.Sequences.Data[i][j])
			if tokenID >= 0 && tokenID < embeddings.Rows {
				for k := 0; k < embeddings.Cols; k++ {
					result.Data[i][k] = embeddings.Data[tokenID][k]
				}
			}
		}
	}
	
	return result
}
