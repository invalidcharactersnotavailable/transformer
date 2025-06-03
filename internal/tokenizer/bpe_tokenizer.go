package transformer

import (
	"bytes"
	"sort"
	"strings"
	"unicode"
)

// BPETokenizer implements Byte-Pair Encoding tokenization
type BPETokenizer struct {
	Vocabulary     map[string]int
	IdToToken      map[int]string
	MergeRules     map[string]int
	SpecialTokens  map[string]int
	UnkToken       string
	BosToken       string
	EosToken       string
	PadToken       string
	MaskToken      string
	MaxTokenLength int
}

// BPETokenizerOptions contains configuration options for BPE tokenizer
type BPETokenizerOptions struct {
	UnkToken       string
	BosToken       string
	EosToken       string
	PadToken       string
	MaskToken      string
	MaxTokenLength int
}

// NewDefaultBPEOptions creates default options for BPE tokenizer
func NewDefaultBPEOptions() *BPETokenizerOptions {
	return &BPETokenizerOptions{
		UnkToken:       "[UNK]",
		BosToken:       "[BOS]",
		EosToken:       "[EOS]",
		PadToken:       "[PAD]",
		MaskToken:      "[MASK]",
		MaxTokenLength: 100,
	}
}

// NewBPETokenizer creates a new BPE tokenizer
func NewBPETokenizer(vocab map[string]int, mergeRules map[string]int, options *BPETokenizerOptions) *BPETokenizer {
	if options == nil {
		options = NewDefaultBPEOptions()
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

	return &BPETokenizer{
		Vocabulary:     vocab,
		IdToToken:      idToToken,
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

// Tokenize splits text into BPE tokens
func (t *BPETokenizer) Tokenize(text string) []string {
	// Normalize text
	text = t.normalizeText(text)
	
	// Split by whitespace first
	words := strings.Fields(text)
	
	// Process each word with BPE
	allTokens := []string{}
	
	for _, word := range words {
		// Start with characters as separate tokens
		chars := []rune(word)
		wordTokens := make([]string, len(chars))
		for i, c := range chars {
			wordTokens[i] = string(c)
		}
		
		// Apply BPE merge operations
		for {
			bestPair := ""
			bestRank := -1
			
			// Find the best pair to merge
			for i := 0; i < len(wordTokens)-1; i++ {
				pair := wordTokens[i] + wordTokens[i+1]
				if rank, exists := t.MergeRules[pair]; exists {
					if bestRank == -1 || rank < bestRank {
						bestPair = pair
						bestRank = rank
					}
				}
			}
			
			// If no more pairs to merge, break
			if bestPair == "" {
				break
			}
			
			// Merge the best pair
			newTokens := []string{}
			i := 0
			for i < len(wordTokens) {
				if i < len(wordTokens)-1 && wordTokens[i]+wordTokens[i+1] == bestPair {
					newTokens = append(newTokens, bestPair)
					i += 2
				} else {
					newTokens = append(newTokens, wordTokens[i])
					i++
				}
			}
			wordTokens = newTokens
		}
		
		allTokens = append(allTokens, wordTokens...)
	}
	
	return allTokens
}

// normalizeText performs text normalization
func (t *BPETokenizer) normalizeText(text string) string {
	// Convert to lowercase
	text = strings.ToLower(text)
	
	// Replace multiple spaces with single space
	text = strings.Join(strings.Fields(text), " ")
	
	return text
}

// Encode converts text to token IDs
func (t *BPETokenizer) Encode(text string, addSpecialTokens bool) []int {
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
func (t *BPETokenizer) Decode(ids []int, skipSpecialTokens bool) string {
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
	
	// Join tokens, removing BPE separators
	return strings.Join(tokens, "").Replace("</w>", " ")
}

// TrainBPE trains a BPE tokenizer from texts
func TrainBPE(texts []string, vocabSize int, options *BPETokenizerOptions) (*BPETokenizer, error) {
	// Count word frequencies
	wordCounts := make(map[string]int)
	
	// First, collect statistics on character pairs
	for _, text := range texts {
		// Normalize and split text
		normalized := strings.ToLower(text)
		words := strings.Fields(normalized)
		
		for _, word := range words {
			// Add end-of-word marker
			word = word + "</w>"
			
			// Count characters
			chars := []rune(word)
			for i := 0; i < len(chars); i++ {
				charStr := string(chars[i])
				wordCounts[charStr]++
			}
			
			// Count character pairs
			for i := 0; i < len(chars)-1; i++ {
				pair := string(chars[i]) + string(chars[i+1])
				wordCounts[pair]++
			}
		}
	}
	
	// Initialize vocabulary with characters
	vocab := make(map[string]int)
	mergeRules := make(map[string]int)
	
	// Add special tokens first
	if options == nil {
		options = NewDefaultBPEOptions()
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
	
	// Add individual characters to vocabulary
	for word, count := range wordCounts {
		if len([]rune(word)) == 1 {
			if _, exists := vocab[word]; !exists {
				vocab[word] = len(vocab)
			}
		}
	}
	
	// Iteratively merge most frequent pairs
	for len(vocab) < vocabSize {
		// Find most frequent pair
		bestPair := ""
		bestCount := 0
		
		for pair, count := range wordCounts {
			if len([]rune(pair)) == 2 {
				if count > bestCount {
					bestPair = pair
					bestCount = count
				}
			}
		}
		
		if bestPair == "" || bestCount == 0 {
			break
		}
		
		// Add to vocabulary and merge rules
		if _, exists := vocab[bestPair]; !exists {
			vocab[bestPair] = len(vocab)
			mergeRules[bestPair] = len(mergeRules)
		}
		
		// Update statistics (simplified)
		wordCounts[bestPair] = 0
	}
	
	return NewBPETokenizer(vocab, mergeRules, options), nil
}
