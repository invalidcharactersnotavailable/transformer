package tokenizer

// SimpleTokenizer represents a basic word-level tokenizer
type SimpleTokenizer struct {
	Vocabulary map[string]int
	IdToToken  map[int]string
	VocabSize  int
}

// NewSimpleTokenizer creates a new simple tokenizer with a given vocabulary
func NewSimpleTokenizer(vocab []string) *SimpleTokenizer {
	vocabulary := make(map[string]int)
	idToToken := make(map[int]string)
	
	for i, token := range vocab {
		vocabulary[token] = i
		idToToken[i] = token
	}
	
	return &SimpleTokenizer{
		Vocabulary: vocabulary,
		IdToToken:  idToToken,
		VocabSize:  len(vocab),
	}
}

// Encode converts a string into token indices
func (t *SimpleTokenizer) Encode(text string) []int {
	// This is a very simple word-level tokenization
	words := []string{}
	currentWord := ""
	
	for _, char := range text {
		if char == ' ' || char == '\n' || char == '\t' {
			if currentWord != "" {
				words = append(words, currentWord)
				currentWord = ""
			}
		} else {
			currentWord += string(char)
		}
	}
	
	if currentWord != "" {
		words = append(words, currentWord)
	}
	
	// Convert words to token indices
	indices := make([]int, len(words))
	for i, word := range words {
		if idx, ok := t.Vocabulary[word]; ok {
			indices[i] = idx
		} else {
			// Use a special unknown token index (0 in this simple implementation)
			indices[i] = 0
		}
	}
	
	return indices
}

// Decode converts token indices back to a string
func (t *SimpleTokenizer) Decode(indices []int) string {
	words := make([]string, len(indices))
	
	for i, idx := range indices {
		if token, ok := t.IdToToken[idx]; ok {
			words[i] = token
		} else {
			words[i] = "<UNK>"
		}
	}
	
	// Join words with spaces
	result := ""
	for i, word := range words {
		if i > 0 {
			result += " "
		}
		result += word
	}
	
	return result
}
