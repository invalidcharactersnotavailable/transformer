package transformer

import (
	"fmt"
	// "sort" // Removed
	"strings"
	"unicode"
	"unicode/utf8"
)

// TokenizerType represents the type of tokenization algorithm to use
type TokenizerType int

const (
	// SimpleTokenization uses basic word-level tokenization
	SimpleTokenization TokenizerType = iota
	// BPETokenization uses Byte-Pair Encoding tokenization
	BPETokenization
	// WordPieceTokenization uses WordPiece tokenization
	WordPieceTokenization
)

// TokenizerOptions contains configuration options for the tokenizer
type TokenizerOptions struct {
	// Special tokens
	UnkToken       string
	BosToken       string
	EosToken       string
	PadToken       string
	MaskToken      string
	SepToken       string
	ClsToken       string
	
	// Configuration
	TokenizerType    TokenizerType
	MaxTokenLength   int
	VocabSize        int
	ModelMaxLength   int
	PaddingSide      string // "right" or "left"
	TruncationSide   string // "right" or "left"
	
	// Processing options
	LowerCase                bool
	StripAccents             bool
	CleanUpTokenizationSpaces bool
	AddPrefixSpace           bool
}

// NewDefaultTokenizerOptions creates default options for the tokenizer
func NewDefaultTokenizerOptions() *TokenizerOptions {
	return &TokenizerOptions{
		// Special tokens
		UnkToken:       "[UNK]",
		BosToken:       "[BOS]",
		EosToken:       "[EOS]",
		PadToken:       "[PAD]",
		MaskToken:      "[MASK]",
		SepToken:       "[SEP]",
		ClsToken:       "[CLS]",
		
		// Configuration
		TokenizerType:    BPETokenization,
		MaxTokenLength:   100,
		VocabSize:        30000,
		ModelMaxLength:   512,
		PaddingSide:      "right",
		TruncationSide:   "right",
		
		// Processing options
		LowerCase:                true,
		StripAccents:             false,
		CleanUpTokenizationSpaces: true,
		AddPrefixSpace:           false,
	}
}

// Tokenizer represents a unified tokenization system supporting multiple strategies
type Tokenizer struct {
	// Core vocabulary
	Vocabulary     map[string]int
	IdToToken      map[int]string
	VocabSize      int
	
	// BPE-specific
	MergeRules     map[string]int
	
	// Special tokens
	SpecialTokens  map[string]int
	SpecialTokensMap map[string]string
	UnkToken       string
	BosToken       string
	EosToken       string
	PadToken       string
	MaskToken      string
	SepToken       string
	ClsToken       string
	
	// Configuration
	TokenizerType  TokenizerType
	MaxTokenLength int
	ModelMaxLength int
	PaddingSide    string
	TruncationSide string
	
	// Processing options
	LowerCase                bool
	StripAccents             bool
	CleanUpTokenizationSpaces bool
	AddPrefixSpace           bool
}

// NewTokenizer creates a new tokenizer with the specified options
func NewTokenizer(vocab map[string]int, options *TokenizerOptions) (*Tokenizer, error) {
	if vocab == nil {
		return nil, fmt.Errorf("vocabulary cannot be nil")
	}
	
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
	specialTokensMap := make(map[string]string)
	
	// Ensure special tokens are in vocabulary
	specialTokensList := []struct {
		name  string
		token string
	}{
		{"unk_token", options.UnkToken},
		{"bos_token", options.BosToken},
		{"eos_token", options.EosToken},
		{"pad_token", options.PadToken},
		{"mask_token", options.MaskToken},
		{"sep_token", options.SepToken},
		{"cls_token", options.ClsToken},
	}
	
	for _, st := range specialTokensList {
		specialTokensMap[st.name] = st.token
		
		if id, exists := vocab[st.token]; exists {
			specialTokens[st.token] = id
		} else {
			// Add to vocabulary with next available ID
			nextID := len(vocab)
			vocab[st.token] = nextID
			idToToken[nextID] = st.token
			specialTokens[st.token] = nextID
		}
	}

	return &Tokenizer{
		// Core vocabulary
		Vocabulary:     vocab,
		IdToToken:      idToToken,
		VocabSize:      len(vocab),
		
		// BPE-specific (initialize empty for non-BPE tokenizers)
		MergeRules:     make(map[string]int),
		
		// Special tokens
		SpecialTokens:  specialTokens,
		SpecialTokensMap: specialTokensMap,
		UnkToken:       options.UnkToken,
		BosToken:       options.BosToken,
		EosToken:       options.EosToken,
		PadToken:       options.PadToken,
		MaskToken:      options.MaskToken,
		SepToken:       options.SepToken,
		ClsToken:       options.ClsToken,
		
		// Configuration
		TokenizerType:  options.TokenizerType,
		MaxTokenLength: options.MaxTokenLength,
		ModelMaxLength: options.ModelMaxLength,
		PaddingSide:    options.PaddingSide,
		TruncationSide: options.TruncationSide,
		
		// Processing options
		LowerCase:                options.LowerCase,
		StripAccents:             options.StripAccents,
		CleanUpTokenizationSpaces: options.CleanUpTokenizationSpaces,
		AddPrefixSpace:           options.AddPrefixSpace,
	}, nil
}

// NewBPETokenizer creates a new BPE tokenizer
func NewBPETokenizer(vocab map[string]int, mergeRules map[string]int, options *TokenizerOptions) (*Tokenizer, error) {
	if options == nil {
		options = NewDefaultTokenizerOptions()
	}
	
	// Set tokenizer type to BPE
	options.TokenizerType = BPETokenization
	
	tokenizer, err := NewTokenizer(vocab, options)
	if err != nil {
		return nil, err
	}
	
	// Add merge rules
	if mergeRules != nil {
		tokenizer.MergeRules = mergeRules
	}
	
	return tokenizer, nil
}

// NewSimpleTokenizer creates a new simple word-level tokenizer
func NewSimpleTokenizer(vocab []string, options *TokenizerOptions) (*Tokenizer, error) {
	if options == nil {
		options = NewDefaultTokenizerOptions()
	}
	
	// Set tokenizer type to Simple
	options.TokenizerType = SimpleTokenization
	
	// Convert vocab slice to map
	vocabMap := make(map[string]int)
	for i, token := range vocab {
		vocabMap[token] = i
	}
	
	return NewTokenizer(vocabMap, options)
}

// Normalize performs Unicode normalization and other text preprocessing
func (t *Tokenizer) Normalize(text string) string {
	// Apply lowercase if configured
	if t.LowerCase {
		text = strings.ToLower(text)
	}
	
	// Replace multiple spaces with single space
	text = strings.Join(strings.Fields(text), " ")
	
	// Strip accents if configured
	if t.StripAccents {
		// This is a simplified implementation
		// In a real implementation, use unicode/norm package
		text = stripAccents(text)
	}
	
	// Add prefix space if configured
	if t.AddPrefixSpace && len(text) > 0 && text[0] != ' ' {
		text = " " + text
	}
	
	return text
}

// stripAccents is a simplified function to remove accents
// In a real implementation, use unicode/norm package
func stripAccents(text string) string {
	result := ""
	for _, r := range text {
		if unicode.Is(unicode.Mn, r) {
			// Skip combining marks
			continue
		}
		result += string(r)
	}
	return result
}

// Tokenize splits text into tokens based on the configured tokenization strategy
func (t *Tokenizer) Tokenize(text string) ([]string, error) {
	if text == "" {
		return []string{}, nil
	}
	
	// Normalize text
	text = t.Normalize(text)
	
	// Apply tokenization based on strategy
	switch t.TokenizerType {
	case SimpleTokenization:
		return t.simpleTokenize(text)
	case BPETokenization:
		return t.bpeTokenize(text)
	case WordPieceTokenization:
		return t.wordpieceTokenize(text)
	default:
		return nil, fmt.Errorf("unknown tokenizer type: %v", t.TokenizerType)
	}
}

// simpleTokenize implements simple word-level tokenization
func (t *Tokenizer) simpleTokenize(text string) ([]string, error) {
	// Split by whitespace
	words := strings.Fields(text)
	
	// Convert words to tokens
	tokens := make([]string, 0, len(words))
	for _, word := range words {
		tokens = append(tokens, word)
	}
	
	return tokens, nil
}

// bpeTokenize implements Byte-Pair Encoding tokenization
func (t *Tokenizer) bpeTokenize(text string) ([]string, error) {
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
	
	return allTokens, nil
}

// wordpieceTokenize implements WordPiece tokenization
func (t *Tokenizer) wordpieceTokenize(text string) ([]string, error) {
	// Split by whitespace first
	words := strings.Fields(text)
	
	// Process each word with WordPiece
	allTokens := []string{}
	
	for _, word := range words {
		// Check if the word is in vocabulary
		if _, exists := t.Vocabulary[word]; exists {
			allTokens = append(allTokens, word)
			continue
		}
		
		// If not in vocabulary, try to split into subwords
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
		
		allTokens = append(allTokens, subTokens...)
	}
	
	return allTokens, nil
}

// EncodeOptions contains options for the Encode method
type EncodeOptions struct {
	AddSpecialTokens      bool
	MaxLength             int
	Truncation            bool
	Padding               bool
	ReturnTokenTypeIds    bool
	ReturnAttentionMask   bool
	ReturnOffsetMapping   bool
	ReturnSpecialTokensMask bool
}

// DefaultEncodeOptions returns the default encoding options
func DefaultEncodeOptions() *EncodeOptions {
	return &EncodeOptions{
		AddSpecialTokens:      true,
		MaxLength:             0, // 0 means no limit
		Truncation:            false,
		Padding:               false,
		ReturnTokenTypeIds:    true,
		ReturnAttentionMask:   true,
		ReturnOffsetMapping:   false,
		ReturnSpecialTokensMask: false,
	}
}

// EncodingResult contains the result of encoding text
type EncodingResult struct {
	InputIds         []int
	TokenTypeIds     []int
	AttentionMask    []int
	SpecialTokensMask []int
	OffsetMapping    [][2]int
	Tokens           []string
}

// Encode converts text to token IDs with various options
func (t *Tokenizer) Encode(text string, options *EncodeOptions) (*EncodingResult, error) {
	if options == nil {
		options = DefaultEncodeOptions()
	}
	
	// Tokenize the text
	tokens, err := t.Tokenize(text)
	if err != nil {
		return nil, err
	}
	
	// Convert tokens to IDs
	inputIds := make([]int, 0, len(tokens))
	specialTokensMask := make([]int, 0, len(tokens))
	
	// Add BOS/CLS token if requested
	if options.AddSpecialTokens {
		if t.TokenizerType == BPETokenization {
			inputIds = append(inputIds, t.SpecialTokens[t.BosToken])
			specialTokensMask = append(specialTokensMask, 1)
		} else {
			inputIds = append(inputIds, t.SpecialTokens[t.ClsToken])
			specialTokensMask = append(specialTokensMask, 1)
		}
	}
	
	// Add token IDs
	for _, token := range tokens {
		if id, exists := t.Vocabulary[token]; exists {
			inputIds = append(inputIds, id)
			specialTokensMask = append(specialTokensMask, 0)
		} else {
			inputIds = append(inputIds, t.SpecialTokens[t.UnkToken])
			specialTokensMask = append(specialTokensMask, 0)
		}
	}
	
	// Add EOS/SEP token if requested
	if options.AddSpecialTokens {
		if t.TokenizerType == BPETokenization {
			inputIds = append(inputIds, t.SpecialTokens[t.EosToken])
			specialTokensMask = append(specialTokensMask, 1)
		} else {
			inputIds = append(inputIds, t.SpecialTokens[t.SepToken])
			specialTokensMask = append(specialTokensMask, 1)
		}
	}
	
	// Apply truncation if needed
	maxLength := options.MaxLength
	if maxLength == 0 {
		maxLength = t.ModelMaxLength
	}
	
	if options.Truncation && len(inputIds) > maxLength {
		if t.TruncationSide == "right" {
			inputIds = inputIds[:maxLength]
			specialTokensMask = specialTokensMask[:maxLength]
		} else {
			inputIds = inputIds[len(inputIds)-maxLength:]
			specialTokensMask = specialTokensMask[len(specialTokensMask)-maxLength:]
		}
	}
	
	// Create attention mask (1 for real tokens, 0 for padding)
	attentionMask := make([]int, len(inputIds))
	for i := range attentionMask {
		attentionMask[i] = 1
	}
	
	// Apply padding if needed
	if options.Padding && maxLength > 0 && len(inputIds) < maxLength {
		padId := t.SpecialTokens[t.PadToken]
		padCount := maxLength - len(inputIds)
		
		if t.PaddingSide == "right" {
			// Pad on the right
			for i := 0; i < padCount; i++ {
				inputIds = append(inputIds, padId)
				attentionMask = append(attentionMask, 0)
				specialTokensMask = append(specialTokensMask, 1)
			}
		} else {
			// Pad on the left
			paddedIds := make([]int, maxLength)
			paddedAttention := make([]int, maxLength)
			paddedSpecial := make([]int, maxLength)
			
			copy(paddedIds[padCount:], inputIds)
			copy(paddedAttention[padCount:], attentionMask)
			copy(paddedSpecial[padCount:], specialTokensMask)
			
			for i := 0; i < padCount; i++ {
				paddedIds[i] = padId
				paddedAttention[i] = 0
				paddedSpecial[i] = 1
			}
			
			inputIds = paddedIds
			attentionMask = paddedAttention
			specialTokensMask = paddedSpecial
		}
	}
	
	// Create token type IDs (all 0 for single sequence)
	tokenTypeIds := make([]int, len(inputIds))
	
	// Create offset mapping (placeholder - would need actual character offsets)
	offsetMapping := make([][2]int, len(inputIds))
	
	// Convert IDs back to tokens for reference
	resultTokens := make([]string, len(inputIds))
	for i, id := range inputIds {
		if token, exists := t.IdToToken[id]; exists {
			resultTokens[i] = token
		} else {
			resultTokens[i] = t.UnkToken
		}
	}
	
	// Create result
	result := &EncodingResult{
		InputIds:         inputIds,
		TokenTypeIds:     tokenTypeIds,
		AttentionMask:    attentionMask,
		SpecialTokensMask: specialTokensMask,
		OffsetMapping:    offsetMapping,
		Tokens:           resultTokens,
	}
	
	return result, nil
}

// BatchEncodeOptions contains options for the BatchEncode method
type BatchEncodeOptions struct {
	EncodeOptions
	ReturnTensors bool // Whether to return tensors instead of lists
}

// DefaultBatchEncodeOptions returns the default batch encoding options
func DefaultBatchEncodeOptions() *BatchEncodeOptions {
	return &BatchEncodeOptions{
		EncodeOptions: *DefaultEncodeOptions(),
		ReturnTensors: false,
	}
}

// BatchEncodingResult contains the result of batch encoding
type BatchEncodingResult struct {
	InputIds         [][]int
	TokenTypeIds     [][]int
	AttentionMask    [][]int
	SpecialTokensMask [][]int
	OffsetMapping    [][][2]int
	Tokens           [][]string
	Lengths          []int
}

// BatchEncode encodes multiple texts with padding
func (t *Tokenizer) BatchEncode(texts []string, options *BatchEncodeOptions) (*BatchEncodingResult, error) {
	if options == nil {
		options = DefaultBatchEncodeOptions()
	}
	
	// Force padding for batch encoding
	options.Padding = true
	
	// First, encode all texts individually
	encodings := make([]*EncodingResult, len(texts))
	maxLen := 0
	
	for i, text := range texts {
		encoding, err := t.Encode(text, &options.EncodeOptions)
		if err != nil {
			return nil, err
		}
		
		encodings[i] = encoding
		if len(encoding.InputIds) > maxLen {
			maxLen = len(encoding.InputIds)
		}
	}
	
	// Set max length if not specified
	if options.MaxLength == 0 {
		options.MaxLength = maxLen
	}
	
	// Re-encode with consistent padding
	for i, text := range texts {
		options.EncodeOptions.MaxLength = options.MaxLength
		encoding, err := t.Encode(text, &options.EncodeOptions)
		if err != nil {
			return nil, err
		}
		encodings[i] = encoding
	}
	
	// Collect results
	result := &BatchEncodingResult{
		InputIds:         make([][]int, len(texts)),
		TokenTypeIds:     make([][]int, len(texts)),
		AttentionMask:    make([][]int, len(texts)),
		SpecialTokensMask: make([][]int, len(texts)),
		OffsetMapping:    make([][][2]int, len(texts)),
		Tokens:           make([][]string, len(texts)),
		Lengths:          make([]int, len(texts)),
	}
	
	for i, encoding := range encodings {
		result.InputIds[i] = encoding.InputIds
		result.TokenTypeIds[i] = encoding.TokenTypeIds
		result.AttentionMask[i] = encoding.AttentionMask
		result.SpecialTokensMask[i] = encoding.SpecialTokensMask
		result.OffsetMapping[i] = encoding.OffsetMapping
		result.Tokens[i] = encoding.Tokens
		result.Lengths[i] = len(encoding.InputIds)
	}
	
	return result, nil
}

// DecodeOptions contains options for the Decode method
type DecodeOptions struct {
	SkipSpecialTokens      bool
	CleanUpTokenizationSpaces bool
}

// DefaultDecodeOptions returns the default decoding options
func DefaultDecodeOptions() *DecodeOptions {
	return &DecodeOptions{
		SkipSpecialTokens:      true,
		CleanUpTokenizationSpaces: true,
	}
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int, options *DecodeOptions) (string, error) {
	if options == nil {
		options = DefaultDecodeOptions()
	}
	
	tokens := make([]string, 0, len(ids))
	
	for _, id := range ids {
		if token, exists := t.IdToToken[id]; exists {
			// Skip special tokens if requested
			if options.SkipSpecialTokens {
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
		} else {
			return "", fmt.Errorf("token ID not found in vocabulary: %d", id)
		}
	}
	
	// Join tokens based on tokenizer type
	var text string
	switch t.TokenizerType {
	case SimpleTokenization:
		text = strings.Join(tokens, " ")
	case BPETokenization:
		// For BPE, join without spaces and then replace special markers
		text = strings.Join(tokens, "")
		text = strings.ReplaceAll(text, "</w>", " ")
	case WordPieceTokenization:
		// For WordPiece, join with spaces and handle continuation tokens
		var builder strings.Builder
		for i, token := range tokens {
			if i > 0 && !strings.HasPrefix(token, "##") {
				builder.WriteString(" ")
			}
			// Remove ## prefix for WordPiece continuation tokens
			token = strings.TrimPrefix(token, "##")
			builder.WriteString(token)
		}
		text = builder.String()
	default:
		return "", fmt.Errorf("unknown tokenizer type: %v", t.TokenizerType)
	}
	
	// Clean up tokenization spaces if requested
	if options.CleanUpTokenizationSpaces {
		text = strings.Join(strings.Fields(text), " ")
	}
	
	return text, nil
}

// BatchDecode decodes multiple token ID sequences
func (t *Tokenizer) BatchDecode(idsList [][]int, options *DecodeOptions) ([]string, error) {
	results := make([]string, len(idsList))
	
	for i, ids := range idsList {
		text, err := t.Decode(ids, options)
		if err != nil {
			return nil, err
		}
		results[i] = text
	}
	
	return results, nil
}

// TrainBPE trains a BPE tokenizer from texts
func TrainBPE(texts []string, vocabSize int, options *TokenizerOptions) (*Tokenizer, error) {
	if options == nil {
		options = NewDefaultTokenizerOptions()
	}
	
	// Set tokenizer type to BPE
	options.TokenizerType = BPETokenization
	options.VocabSize = vocabSize
	
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
	specialTokens := []string{
		options.UnkToken,
		options.BosToken,
		options.EosToken,
		options.PadToken,
		options.MaskToken,
		options.SepToken,
		options.ClsToken,
	}
	
	for i, token := range specialTokens {
		vocab[token] = i
	}
	
	// Add individual characters to vocabulary
	for word, _ := range wordCounts { // count replaced with _
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
	
	return NewBPETokenizer(vocab, mergeRules, options)
}

// SaveVocabulary saves the tokenizer vocabulary to a file
func (t *Tokenizer) SaveVocabulary(path string) error {
	// Implementation would write vocabulary to file
	// This is a placeholder
	return fmt.Errorf("SaveVocabulary not implemented")
}

// LoadVocabulary loads the tokenizer vocabulary from a file
func LoadVocabulary(path string, options *TokenizerOptions) (*Tokenizer, error) {
	// Implementation would read vocabulary from file
	// This is a placeholder
	return nil, fmt.Errorf("LoadVocabulary not implemented")
}
