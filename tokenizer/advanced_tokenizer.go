package transformer

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// AdvancedTokenizer represents a subword tokenization system
type AdvancedTokenizer struct {
	Vocabulary     map[string]int
	IdToToken      map[int]string
	VocabSize      int
	MergeRules     map[string]int
	SpecialTokens  map[string]int
	UnkToken       string
	BosToken       string
	EosToken       string
	PadToken       string
	MaskToken      string
	MaxTokenLength int
}

// TokenizerOptions contains configuration options for the tokenizer
type TokenizerOptions struct {
	UnkToken       string
	BosToken       string
	EosToken       string
	PadToken       string
	MaskToken      string
	MaxTokenLength int
}

// NewDefaultTokenizerOptions creates default options for the tokenizer
func NewDefaultTokenizerOptions() *TokenizerOptions {
	return &TokenizerOptions{
		UnkToken:       "[UNK]",
		BosToken:       "[BOS]",
		EosToken:       "[EOS]",
		PadToken:       "[PAD]",
		MaskToken:      "[MASK]",
		MaxTokenLength: 100,
	}
}

// NewAdvancedTokenizer creates a new advanced tokenizer
func NewAdvancedTokenizer(vocab map[string]int, mergeRules map[string]int, options *TokenizerOptions) *AdvancedTokenizer {
	if options == nil {
		options = NewDefaultTokenizerOptions()
	}

	// Create reverse mapping
	idToToken := make(map[int]string)
	for token, id := range vocab {
		idToToken[id] = token
	}

	// Add special tokens if they don't exist
	specialTokens := make(map[string]int)
	
	// Ensure special tokens are in vocabulary
	for _, token := range []string{
		options.UnkToken,
		options.BosToken,
		options.EosToken,
		options.PadToken,
		options.MaskToken,
	} {
		if id, exists := vocab[token]; exists {
			specialTokens[token] = id
		} else {
			// Add to vocabulary with next available ID
			nextID := len(vocab)
			vocab[token] = nextID
			idToToken[nextID] = token
			specialTokens[token] = nextID
		}
	}

	return &AdvancedTokenizer{
		Vocabulary:     vocab,
		IdToToken:      idToToken,
		VocabSize:      len(vocab),
		MergeRules:     mergeRules,
		SpecialTokens:  specialTokens,
		UnkToken:       options.UnkToken,
		BosToken:       options.BosToken,
		EosToken:       options.EosToken,
		PadToken:       options.PadToken,
		MaskToken:      options.MaskToken,
		MaxTokenLength: options.MaxTokenLength,
	}
}

// Normalize performs Unicode normalization and other text preprocessing
func (t *AdvancedTokenizer) Normalize(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Replace multiple spaces with single space
	text = strings.Join(strings.Fields(text), " ")
	
	// Simple Unicode normalization (in a real implementation, use unicode/norm package)
	// This is a placeholder for actual normalization
	
	return text
}

// Tokenize splits text into subword tokens
func (t *AdvancedTokenizer) Tokenize(text string) []string {
	// Normalize text
	text = t.Normalize(text)
	
	// Initial character-level tokenization
	tokens := []string{}
	
	// Split by whitespace first
	words := strings.Fields(text)
	
	for _, word := range words {
		// Apply byte-pair encoding or wordpiece tokenization
		// This is a simplified version - a real implementation would use the merge rules
		
		// Check if the word is in vocabulary
		if _, exists := t.Vocabulary[word]; exists {
			tokens = append(tokens, word)
			continue
		}
		
		// If not in vocabulary, try to split into subwords
		// This is a simplified greedy algorithm
		start := 0
		subTokens := []string{}
		
		for start < len(word) {
			end := len(word)
			foundSubword := false
			
			// Try to find the longest subword
			for end > start {
				subword := word[start:end]
				if _, exists := t.Vocabulary[subword]; exists {
					subTokens = append(subTokens, subword)
					start = end
					foundSubword = true
					break
				}
				end--
			}
			
			// If no subword found, add a character and continue
			if !foundSubword {
				// Get the next rune
				r, size := utf8.DecodeRuneInString(word[start:])
				if r == utf8.RuneError {
					// Handle invalid UTF-8
					subTokens = append(subTokens, t.UnkToken)
					start += size
				} else {
					subTokens = append(subTokens, string(r))
					start += size
				}
			}
		}
		
		tokens = append(tokens, subTokens...)
	}
	
	return tokens
}

// Encode converts text to token IDs
func (t *AdvancedTokenizer) Encode(text string, addSpecialTokens bool) []int {
	tokens := t.Tokenize(text)
	ids := make([]int, 0, len(tokens))
	
	// Add BOS token if requested
	if addSpecialTokens {
		ids = append(ids, t.SpecialTokens[t.BosToken])
	}
	
	// Convert tokens to IDs
	for _, token := range tokens {
		if id, exists := t.Vocabulary[token]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.SpecialTokens[t.UnkToken])
		}
	}
	
	// Add EOS token if requested
	if addSpecialTokens {
		ids = append(ids, t.SpecialTokens[t.EosToken])
	}
	
	return ids
}

// Decode converts token IDs back to text
func (t *AdvancedTokenizer) Decode(ids []int, skipSpecialTokens bool) string {
	tokens := make([]string, 0, len(ids))
	
	for _, id := range ids {
		if token, exists := t.IdToToken[id]; exists {
			// Skip special tokens if requested
			if skipSpecialTokens {
				isSpecial := false
				for _, specialID := range t.SpecialTokens {
					if id == specialID {
						isSpecial = true
						break
					}
				}
				if isSpecial {
					continue
				}
			}
			tokens = append(tokens, token)
		}
	}
	
	// Join tokens with spaces
	// In a real implementation, this would handle subword merging properly
	return strings.Join(tokens, " ")
}

// BatchEncode encodes multiple texts with padding
func (t *AdvancedTokenizer) BatchEncode(texts []string, addSpecialTokens bool, padToMaxLength bool) ([][]int, []int) {
	encodings := make([][]int, len(texts))
	attentionMasks := make([][]int, len(texts))
	
	// First, encode all texts
	maxLen := 0
	for i, text := range texts {
		encodings[i] = t.Encode(text, addSpecialTokens)
		if len(encodings[i]) > maxLen {
			maxLen = len(encodings[i])
		}
	}
	
	// Apply padding if requested
	if padToMaxLength {
		padID := t.SpecialTokens[t.PadToken]
		
		for i := range encodings {
			attentionMasks[i] = make([]int, maxLen)
			
			// Set attention mask (1 for real tokens, 0 for padding)
			for j := 0; j < len(encodings[i]); j++ {
				attentionMasks[i][j] = 1
			}
			
			// Add padding
			if len(encodings[i]) < maxLen {
				padding := make([]int, maxLen-len(encodings[i]))
				for j := range padding {
					padding[j] = padID
				}
				encodings[i] = append(encodings[i], padding...)
			}
		}
	}
	
	// Return the lengths for creating attention masks
	lengths := make([]int, len(texts))
	for i, enc := range encodings {
		lengths[i] = len(enc)
	}
	
	return encodings, lengths
}

// BuildVocabularyFromTexts builds a vocabulary from a corpus of texts
func BuildVocabularyFromTexts(texts []string, vocabSize int, options *TokenizerOptions) (*AdvancedTokenizer, error) {
	// Count word frequencies
	wordCounts := make(map[string]int)
	
	for _, text := range texts {
		// Normalize and split text
		normalized := strings.ToLower(text)
		words := strings.Fields(normalized)
		
		for _, word := range words {
			// Count characters and character pairs for BPE
			chars := []rune(word)
			for i := 0; i < len(chars); i++ {
				charStr := string(chars[i])
				wordCounts[charStr]++
			}
			
			for i := 0; i < len(chars)-1; i++ {
				pair := string(chars[i]) + string(chars[i+1])
				wordCounts[pair]++
			}
		}
	}
	
	// Sort by frequency
	type wordCount struct {
		word  string
		count int
	}
	
	sortedCounts := make([]wordCount, 0, len(wordCounts))
	for word, count := range wordCounts {
		sortedCounts = append(sortedCounts, wordCount{word, count})
	}
	
	sort.Slice(sortedCounts, func(i, j int) bool {
		return sortedCounts[i].count > sortedCounts[j].count
	})
	
	// Build vocabulary
	vocab := make(map[string]int)
	mergeRules := make(map[string]int)
	
	// Add special tokens first
	if options == nil {
		options = NewDefaultTokenizerOptions()
	}
	
	specialTokens := []string{
		options.UnkToken,
		options.BosToken,
		options.EosToken,
		options.PadToken,
		options.MaskToken,
	}
	
	for i, token := range specialTokens {
		vocab[token] = i
	}
	
	// Add most frequent tokens
	for i, wc := range sortedCounts {
		if len(vocab) >= vocabSize {
			break
		}
		
		if _, exists := vocab[wc.word]; !exists {
			vocab[wc.word] = len(vocab)
			
			// Add merge rule for pairs
			if len(wc.word) == 2 {
				mergeRules[wc.word] = i
			}
		}
	}
	
	return NewAdvancedTokenizer(vocab, mergeRules, options), nil
}
