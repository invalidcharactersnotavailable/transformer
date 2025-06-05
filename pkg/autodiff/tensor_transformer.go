package autodiff

import (
	"fmt"
	"math"
	"math/rand"

	"transformer/pkg/core"
	// MoE types are now in the autodiff package
)

// TransformerWithTensors structure
type TransformerWithTensors struct {
	Encoder           []*EncoderLayerWithTensors
	Decoder           []*DecoderLayerWithTensors
	PositionalEncoder *PositionalEncodingTensor
	EmbeddingLayer    *EmbeddingTensor
	OutputMatrix      *Tensor
	OutputBias        *Tensor
	Config            *core.Config
	Graph             *ComputationGraph
	Dropout           *DropoutTensor
}

// NewTransformerWithTensors constructor
func NewTransformerWithTensors(config *core.Config, graph *ComputationGraph) *TransformerWithTensors {
	if graph == nil { graph = NewComputationGraph() }
	requiresGrad := true

	// Populate MoELayerConfig from core.Config
	// Note: RouterZLossCoeff and LoadBalanceLossCoeff are now part of MoELayerConfig itself.
	// They will be passed from TensorFineTuningConfig to core.Config, then to MoELayerConfig.
	moeLayerConf := MoELayerConfig{ // moe. prefix removed
		ModelDim:             config.EmbeddingDim,
		NumExperts:           config.MoENumExperts,
		HiddenDim:            config.MoEHiddenDim,
		TopK:                 config.MoETopK,
		CapacityFactor:       config.MoECapacityFactor,
		NoisyRouting:         config.MoENoisyRouting,
		RouterZLossCoeff:     config.MoERouterZLossCoeff,     // Get from core.Config
		LoadBalanceLossCoeff: config.MoELoadBalanceLossCoeff, // Get from core.Config
	}
	if moeActivationName := config.MoEActivationName; moeActivationName != "" {
		switch moeActivationName {
		case "gelu":
			moeLayerConf.Activation = GELU
		case "relu":
			moeLayerConf.Activation = ReLU
		default:
			fmt.Printf("Warning: Unknown MoE activation name '%s', defaulting to GELU.\n", moeActivationName)
			moeLayerConf.Activation = GELU
		}
	} else {
		moeLayerConf.Activation = GELU // Default if not specified
	}


	encoder := make([]*EncoderLayerWithTensors, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		if config.UseCrossLayerParameterSharing && i > 0 { encoder[i] = encoder[0]
		} else { encoder[i] = NewEncoderLayerWithTensors(config, config.DropoutRate, config.UseMoE, moeLayerConf, requiresGrad, graph) }
	}
	decoder := make([]*DecoderLayerWithTensors, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		if config.UseCrossLayerParameterSharing && i > 0 { decoder[i] = decoder[0]
		} else { decoder[i] = NewDecoderLayerWithTensors(config, config.DropoutRate, config.UseMoE, moeLayerConf, requiresGrad, graph) }
	}
	
	embeddingL := NewEmbeddingTensor(config.VocabSize,config.EmbeddingDim,requiresGrad,graph)
	posEncL := NewPositionalEncodingTensor(config.EmbeddingDim,config.MaxLen,graph)
	outMatCfg := &TensorConfig{RequiresGrad:true, Name:"output_matrix", Graph:graph}
	outMat,_ := NewRandomTensor(config.EmbeddingDim, config.VocabSize, outMatCfg)
	outBiasData, _ := NewMatrix(1, config.VocabSize)
	outBiasCfg := &TensorConfig{RequiresGrad:true, Name:"output_bias", Graph:graph}
	outBias,_ := NewTensor(outBiasData, outBiasCfg)

	return &TransformerWithTensors{ Encoder:encoder, Decoder:decoder, PositionalEncoder:posEncL, EmbeddingLayer:embeddingL, OutputMatrix:outMat, OutputBias:outBias, Config:config, Graph:graph, Dropout:NewDropoutTensor(config.DropoutRate) }
}

// EncoderLayerWithTensors structure and constructor
type EncoderLayerWithTensors struct {
	SelfAttention *MultiHeadAttentionWithTensors; FeedForward *FeedForwardWithTensors
	Norm1 *LayerNormWithTensors; Norm2 *LayerNormWithTensors; Dropout *DropoutTensor
	MoELayer *MoELayer; IsMoE bool; Graph *ComputationGraph // moe. prefix removed
}
func NewEncoderLayerWithTensors(config *core.Config, dropoutRate float64, useMoE bool, moeConfig MoELayerConfig, requiresGrad bool, graph *ComputationGraph) *EncoderLayerWithTensors { // moe. prefix removed
	el := &EncoderLayerWithTensors{
		SelfAttention: NewMultiHeadAttentionWithTensors(config.EmbeddingDim, config.NumHeads, dropoutRate, requiresGrad, graph),
		Norm1:         NewLayerNormWithTensors(config.EmbeddingDim, requiresGrad, graph),
		Norm2:         NewLayerNormWithTensors(config.EmbeddingDim, requiresGrad, graph),
		Dropout:       NewDropoutTensor(dropoutRate), IsMoE: useMoE, Graph: graph,
	}

	var ffActivationFunc func(*Tensor) (*Tensor, error)
	switch config.ActivationFuncName {
	case "relu":
		ffActivationFunc = ReLU
	case "gelu":
		ffActivationFunc = GELU
	default:
		fmt.Printf("Warning: Unknown FFN activation name '%s', defaulting to GELU.\n", config.ActivationFuncName)
		ffActivationFunc = GELU
	}

	if useMoE {
		moeConfig.ModelDim = config.EmbeddingDim
		if moeConfig.Activation == nil { // If not already set by TransformerWithTensors constructor from MoEActivationName
			// Default MoE expert activation can be same as standard FFN or specific
			moeConfig.Activation = ffActivationFunc // Or a specific MoE default like GELU
		}
		el.MoELayer = NewMoELayer(moeConfig, requiresGrad, graph); el.FeedForward = nil // moe. prefix removed
	} else {
		el.FeedForward = NewFeedForwardWithTensors(config.EmbeddingDim, config.FFNHiddenDim, dropoutRate, ffActivationFunc, requiresGrad, graph); el.MoELayer = nil
	}
	return el
}

// DecoderLayerWithTensors structure and constructor
type DecoderLayerWithTensors struct {
	SelfAttention *MultiHeadAttentionWithTensors; CrossAttention *MultiHeadAttentionWithTensors
	FeedForward *FeedForwardWithTensors; Norm1 *LayerNormWithTensors; Norm2 *LayerNormWithTensors; Norm3 *LayerNormWithTensors
	Dropout *DropoutTensor; MoELayer *MoELayer; IsMoE bool; Graph *ComputationGraph // moe. prefix removed
}
func NewDecoderLayerWithTensors(config *core.Config, dropoutRate float64, useMoE bool, moeConfig MoELayerConfig, requiresGrad bool, graph *ComputationGraph) *DecoderLayerWithTensors { // moe. prefix removed
	dl := &DecoderLayerWithTensors{
		SelfAttention: NewMultiHeadAttentionWithTensors(config.EmbeddingDim,config.NumHeads,dropoutRate,requiresGrad,graph), CrossAttention: NewMultiHeadAttentionWithTensors(config.EmbeddingDim,config.NumHeads,dropoutRate,requiresGrad,graph),
		Norm1: NewLayerNormWithTensors(config.EmbeddingDim,requiresGrad,graph), Norm2: NewLayerNormWithTensors(config.EmbeddingDim,requiresGrad,graph), Norm3: NewLayerNormWithTensors(config.EmbeddingDim,requiresGrad,graph),
		Dropout: NewDropoutTensor(dropoutRate), IsMoE: useMoE, Graph: graph,
	}
	var ffActivationFunc func(*Tensor) (*Tensor, error)
	switch config.ActivationFuncName {
	case "relu":
		ffActivationFunc = ReLU
	case "gelu":
		ffActivationFunc = GELU
	default:
		ffActivationFunc = GELU
	}

	if useMoE {
		moeConfig.ModelDim = config.EmbeddingDim
		if moeConfig.Activation == nil { moeConfig.Activation = ffActivationFunc }
		dl.MoELayer = NewMoELayer(moeConfig, requiresGrad, graph); dl.FeedForward = nil // moe. prefix removed
	} else {
		dl.FeedForward = NewFeedForwardWithTensors(config.EmbeddingDim,config.FFNHiddenDim,dropoutRate,ffActivationFunc,requiresGrad,graph); dl.MoELayer = nil
	}
	return dl
}

// MHA structure and constructor (assuming unchanged from previous correct version)
type MultiHeadAttentionWithTensors struct {
	NumHeads int; ModelDim int; HeadDim int; QueryWeight *Tensor; KeyWeight *Tensor; ValueWeight *Tensor; OutputWeight *Tensor
	AttentionDropout *DropoutTensor; Graph *ComputationGraph
}
func NewMultiHeadAttentionWithTensors(modelDim,numHeads int,attDropRate float64,reqGrad bool,g *ComputationGraph) *MultiHeadAttentionWithTensors {
	hD:=modelDim/numHeads;sfx:=fmt.Sprintf("_mha_d%d_h%d",modelDim,numHeads); tc:=func(n string)*TensorConfig{return &TensorConfig{RequiresGrad:reqGrad,Name:n,Graph:g}}
	return &MultiHeadAttentionWithTensors{NumHeads:numHeads,ModelDim:modelDim,HeadDim:hD,Graph:g,
		QueryWeight:NewRandomTensorFallback(modelDim,modelDim,tc("q_w"+sfx)), KeyWeight:NewRandomTensorFallback(modelDim,modelDim,tc("k_w"+sfx)),
		ValueWeight:NewRandomTensorFallback(modelDim,modelDim,tc("v_w"+sfx)), OutputWeight:NewRandomTensorFallback(modelDim,modelDim,tc("o_w"+sfx)),
		AttentionDropout:NewDropoutTensor(attDropRate),
	}
}

// FFN structure and constructor (assuming unchanged from previous correct version)
type FeedForwardWithTensors struct {
	InputDim int; HiddenDim int; W1 *Tensor; B1 *Tensor; W2 *Tensor; B2 *Tensor
	Dropout *DropoutTensor; Graph *ComputationGraph; Activation func(*Tensor)(*Tensor,error)
}
func NewFeedForwardWithTensors(inD,hidD int,dropRate float64,act func(*Tensor)(*Tensor,error),reqGrad bool,g *ComputationGraph) *FeedForwardWithTensors {
	sfx:=fmt.Sprintf("_ffn_i%d_h%d",inD,hidD);tc:=func(n string)*TensorConfig{return &TensorConfig{RequiresGrad:reqGrad,Name:n,Graph:g}}; a:=act;if a==nil{a=GELU} // Ensure 'a' gets GELU if act is nil
	return &FeedForwardWithTensors{InputDim:inD,HiddenDim:hidD,Graph:g,Activation:a,Dropout:NewDropoutTensor(dropRate),
		W1:NewRandomTensorFallback(inD,hidD,tc("w1"+sfx)), B1:NewZerosTensorFallback(nil,1,hidD,tc("b1"+sfx)),
		W2:NewRandomTensorFallback(hidD,inD,tc("w2"+sfx)), B2:NewZerosTensorFallback(nil,1,inD,tc("b2"+sfx)),
	}
}

// LayerNorm structure and constructor (assuming unchanged from previous correct version)
type LayerNormWithTensors struct { Dim int; Gamma *Tensor; Beta *Tensor; Eps float64; Graph *ComputationGraph }
func NewLayerNormWithTensors(dim int,reqGrad bool,g *ComputationGraph) *LayerNormWithTensors {
	sfx:=fmt.Sprintf("_ln_d%d",dim);tcG:=&TensorConfig{RequiresGrad:reqGrad,Name:"gamma"+sfx,Graph:g};tcB:=&TensorConfig{RequiresGrad:reqGrad,Name:"beta"+sfx,Graph:g}
	gD,_:=NewMatrix(1,dim);for j:=0;j<dim;j++{gD.Data[0][j]=1.0};gam,_:=NewTensor(gD,tcG)
	bD,_:=NewMatrix(1,dim);bet,_:=NewTensor(bD,tcB)
	return &LayerNormWithTensors{Dim:dim,Gamma:gam,Beta:bet,Eps:1e-5,Graph:g}
}

// PositionalEncodingTensor structure and constructor (assuming unchanged)
type PositionalEncodingTensor struct { Dim int; MaxLen int; Encoding *Tensor; Graph *ComputationGraph }
func NewPositionalEncodingTensor(dim,maxLen int,g *ComputationGraph) *PositionalEncodingTensor {
	dat:=make([][]float64,maxLen);for p:=0;p<maxLen;p++{dat[p]=make([]float64,dim);for i:=0;i<dim;i+=2{den:=math.Pow(10000,float64(i)/float64(dim));if i<dim{dat[p][i]=math.Sin(float64(p)/den)};if i+1<dim{dat[p][i+1]=math.Cos(float64(p)/den)}}}
	mat,_:=NewMatrix(maxLen,dim,dat...);pet,_:=NewTensor(mat,&TensorConfig{RequiresGrad:false,Name:"pe_const",Graph:g})
	return &PositionalEncodingTensor{Dim:dim,MaxLen:maxLen,Encoding:pet,Graph:g}
}

// Forward for PositionalEncodingTensor (assuming unchanged)
func (pe *PositionalEncodingTensor) Forward(embeddings *Tensor) (*Tensor, error) {
	numTokens := embeddings.Shape()[0]; sliceEnd := numTokens;
	if numTokens > pe.MaxLen { sliceEnd = pe.MaxLen; fmt.Printf("Warning: PE input seqLen %d > MaxLen %d.\n", numTokens, pe.MaxLen) }
	slicedPE, err := TensorSlice(pe.Encoding, []*SliceArg{{Start:0, End: sliceEnd},{Start:0, End:pe.Dim}}, "pe_slice")
	if err != nil { return nil, fmt.Errorf("slicing PE: %w", err)}
	if numTokens > pe.MaxLen {
		embeddingsPrefix, errSlice := TensorSlice(embeddings, []*SliceArg{{Start:0,End:pe.MaxLen},{Start:0,End:embeddings.Shape()[1]}}, "emb_prefix_for_pe")
		if errSlice != nil { return nil, fmt.Errorf("slicing embeddings for PE: %w", errSlice)}
		// fmt.Printf("Warning: Output sequence effectively truncated to PE.MaxLen %d from %d tokens due to PE.\n", pe.MaxLen, numTokens) // Reduce verbosity
		return Add(embeddingsPrefix, slicedPE)
	}
	return Add(embeddings, slicedPE)
}

// EmbeddingTensor structure and constructor (assuming unchanged)
type EmbeddingTensor struct { NumEmbeddings int; EmbeddingDim int; Weights *Tensor; Graph *ComputationGraph }
func NewEmbeddingTensor(numEmb,embDim int,reqGrad bool,g *ComputationGraph) *EmbeddingTensor {
	w,_:=NewRandomTensor(numEmb,embDim,&TensorConfig{RequiresGrad:reqGrad,Name:"emb_weights",Graph:g})
	return &EmbeddingTensor{NumEmbeddings:numEmb,EmbeddingDim:embDim,Weights:w,Graph:g}
}

// Forward for EmbeddingTensor (assuming unchanged from previous correct version using TensorGather)
func (e *EmbeddingTensor) Forward(indices *Tensor) (*Tensor, error) {
	if indices.Data == nil { return nil, fmt.Errorf("indices tensor data is nil") }
	batchSize := indices.Shape()[0]; seqLen := indices.Shape()[1]; numIndicesTotal := batchSize * seqLen
	if numIndicesTotal == 0 {
		emptyData, _ := NewMatrix(0, e.EmbeddingDim)
		return NewTensor(emptyData,&TensorConfig{Graph:e.Graph,RequiresGrad:e.Weights.RequiresGrad, Name:fmt.Sprintf("EmbedEmpty(%s)",indices.Name)})
	}
	flatIndicesData := make([][]float64, numIndicesTotal)
	idxCounter := 0
	for i := 0; i < batchSize; i++ { for j := 0; j < seqLen; j++ { flatIndicesData[idxCounter] = []float64{indices.Data.Data[i][j]}; idxCounter++ } }
	flatIndicesMatrix, err := NewMatrix(numIndicesTotal, 1, flatIndicesData...); if err != nil { return nil, err}
	flatIndicesTensor, err := NewTensor(flatIndicesMatrix, &TensorConfig{Graph: e.Graph, Name: fmt.Sprintf("FlatIndices_Embed(%s)", indices.Name), RequiresGrad: false}); if err != nil { return nil, err}
	gatheredEmbeddings, err := TensorGather(e.Weights, flatIndicesTensor, 0)
	if err != nil { return nil, fmt.Errorf("embedding lookup via TensorGather failed: %w", err) }
	gatheredEmbeddings.Name = fmt.Sprintf("EmbedOut(%s)", indices.Name)
	return gatheredEmbeddings, nil
}

// Forward for TransformerWithTensors (assuming unchanged from previous correct version)
func (t *TransformerWithTensors) Forward(srcIndicesTensor, tgtIndicesTensor *Tensor, srcMask, tgtMask *Tensor, isTraining bool) (*Tensor, error) {
	var err error
	srcEmbedded, err := t.EmbeddingLayer.Forward(srcIndicesTensor); if err != nil { return nil, fmt.Errorf("src embed: %w", err) }
	tgtEmbedded, err := t.EmbeddingLayer.Forward(tgtIndicesTensor); if err != nil { return nil, fmt.Errorf("tgt embed: %w", err) }
	srcWithPos, err := t.PositionalEncoder.Forward(srcEmbedded); if err != nil { return nil, fmt.Errorf("src posenc: %w", err) }
	tgtWithPos, err := t.PositionalEncoder.Forward(tgtEmbedded); if err != nil { return nil, fmt.Errorf("tgt posenc: %w", err) }
	currentSrc := srcWithPos; currentTgt := tgtWithPos
	if t.Dropout != nil {
		currentSrc, err = t.Dropout.Forward(currentSrc, isTraining); if err != nil { return nil, fmt.Errorf("src dropout: %w", err)}
		currentTgt, err = t.Dropout.Forward(currentTgt, isTraining); if err != nil { return nil, fmt.Errorf("tgt dropout: %w", err)}
	}
	encoderOutput := currentSrc
	for i, layer := range t.Encoder { encoderOutput, err = layer.Forward(encoderOutput, isTraining); if err != nil { return nil, fmt.Errorf("enc layer %d: %w", i, err) } }
	decoderOutput := currentTgt
	for i, layer := range t.Decoder { decoderOutput, err = layer.Forward(decoderOutput, encoderOutput, isTraining, srcMask, tgtMask); if err != nil { return nil, fmt.Errorf("dec layer %d: %w", i, err) } }
	logits, err := MatMul(decoderOutput, t.OutputMatrix); if err != nil { return nil, fmt.Errorf("final MatMul: %w", err) }
	if t.OutputBias != nil { logits, err = Add(logits, t.OutputBias); if err != nil { return nil, fmt.Errorf("output bias: %w", err)} }
	return logits, nil
}

// GetMoELayers and GetParameters methods (assuming unchanged)
func (t *TransformerWithTensors) GetMoELayers() []*MoELayer { /* ... */ // moe. prefix removed
	ls := []*MoELayer{}; for _, l := range t.Encoder { if l.IsMoE && l.MoELayer != nil { ls = append(ls, l.MoELayer) } }; for _, l := range t.Decoder { if l.IsMoE && l.MoELayer != nil { ls = append(ls, l.MoELayer) } }; return ls // moe. prefix removed
}
func (t *TransformerWithTensors) GetParameters() []*Tensor { /* ... */
	pm:=make(map[*Tensor]bool);ap:=[]*Tensor{};addP:=func(p*Tensor){if p!=nil&&p.RequiresGrad&&!pm[p]{ap=append(ap,p);pm[p]=true}};addPS:=func(ps[]*Tensor){for _,p:=range ps{addP(p)}}
	if t.EmbeddingLayer!=nil{addPS(t.EmbeddingLayer.GetParameters())};addP(t.OutputMatrix);addP(t.OutputBias)
	for _,l:=range t.Encoder{addPS(l.GetParameters())};for _,l:=range t.Decoder{addPS(l.GetParameters())}; return ap
}
func (t *TransformerWithTensors) GetNamedParameters() map[string]*Tensor { /* ... */
	p:=make(map[string]*Tensor)
	if t.EmbeddingLayer!=nil&&t.EmbeddingLayer.Weights!=nil&&t.EmbeddingLayer.Weights.RequiresGrad{p[t.EmbeddingLayer.Weights.Name]=t.EmbeddingLayer.Weights}
	if t.OutputMatrix!=nil&&t.OutputMatrix.RequiresGrad{p[t.OutputMatrix.Name]=t.OutputMatrix}
	if t.OutputBias!=nil&&t.OutputBias.RequiresGrad{p[t.OutputBias.Name]=t.OutputBias}
	addNP:=func(lps[]*Tensor,prfx string,idx int,shrd bool){for _,param:=range lps{if param!=nil&&param.RequiresGrad{n:=param.Name;if n==""{n=fmt.Sprintf("unnamed_%s_%d_p%p",prfx,idx,param)};fn:=fmt.Sprintf("%s_%d_%s",prfx,idx,n);if shrd{fn=fmt.Sprintf("shared_%s_%s",prfx,n)};p[fn]=param}}}
	if t.Config.UseCrossLayerParameterSharing{if len(t.Encoder)>0{addNP(t.Encoder[0].GetParameters(),"encoder",0,true)};if len(t.Decoder)>0{addNP(t.Decoder[0].GetParameters(),"decoder",0,true)}}else{for i,l:=range t.Encoder{addNP(l.GetParameters(),"encoder",i,false)};for i,l:=range t.Decoder{addNP(l.GetParameters(),"decoder",i,false)}}
	return p
}

// Fallbacks and Component GetParameters (assuming unchanged)
// NewRandomTensorFallback and NewZerosTensorFallback are defined in moe_components.go (package autodiff)
func (e *EmbeddingTensor) GetParameters() []*Tensor { if e.Weights!=nil&&e.Weights.RequiresGrad{return[]*Tensor{e.Weights}}; return[]*Tensor{} }
func (pe *PositionalEncodingTensor) GetParameters() []*Tensor { return []*Tensor{} }
func (el *EncoderLayerWithTensors) GetParameters() []*Tensor { ps:=el.SelfAttention.GetParameters();if el.IsMoE{if el.MoELayer!=nil{ps=append(ps,el.MoELayer.GetParameters()...)}}else{if el.FeedForward!=nil{ps=append(ps,el.FeedForward.GetParameters()...)}};ps=append(ps,el.Norm1.GetParameters()...);ps=append(ps,el.Norm2.GetParameters()...);return ps }
func (dl *DecoderLayerWithTensors) GetParameters() []*Tensor { ps:=dl.SelfAttention.GetParameters();ps=append(ps,dl.CrossAttention.GetParameters()...);if dl.IsMoE{if dl.MoELayer!=nil{ps=append(ps,dl.MoELayer.GetParameters()...)}}else{if dl.FeedForward!=nil{ps=append(ps,dl.FeedForward.GetParameters()...)}};ps=append(ps,dl.Norm1.GetParameters()...);ps=append(ps,dl.Norm2.GetParameters()...);ps=append(ps,dl.Norm3.GetParameters()...);return ps }
func (mha *MultiHeadAttentionWithTensors) GetParameters() []*Tensor { return []*Tensor{mha.QueryWeight,mha.KeyWeight,mha.ValueWeight,mha.OutputWeight} }
func (ff *FeedForwardWithTensors) GetParameters() []*Tensor { return []*Tensor{ff.W1,ff.B1,ff.W2,ff.B2} }
func (ln *LayerNormWithTensors) GetParameters() []*Tensor { return []*Tensor{ln.Gamma,ln.Beta} }

// DropoutTensor struct and methods (assuming unchanged)
type DropoutTensor struct { Rate float64 }
func NewDropoutTensor(rate float64) *DropoutTensor { return &DropoutTensor{Rate: rate} }
func (d *DropoutTensor) Forward(input *Tensor, isTraining bool) (*Tensor, error) {
    if !isTraining || d.Rate == 0 { return input, nil }
    return DropoutTensor(input, d.Rate, isTraining, fmt.Sprintf("%s_do",input.Name) )
}

// Layer Forward methods (assuming structure largely unchanged, but calls graph ops)
func (el *EncoderLayerWithTensors) Forward(input *Tensor, isTraining bool) (*Tensor, error) {
	var err error; var attnOut, norm1Out, residual1, norm_output_for_ffn, sublayer_output, finalResidual *Tensor
	norm1Out, err = el.Norm1.Forward(input); if err != nil { return nil, fmt.Errorf("enc Norm1: %w", err) }
	attnOut, err = el.SelfAttention.Forward(norm1Out, norm1Out, norm1Out, nil, isTraining); if err != nil { return nil, fmt.Errorf("enc SelfAttn: %w", err) }
	attnOut, err = el.Dropout.Forward(attnOut, isTraining); if err != nil { return nil, fmt.Errorf("enc AttnDropout: %w", err) }
	residual1, err = Add(input, attnOut); if err != nil { return nil, fmt.Errorf("enc Res1: %w", err) }
	norm_output_for_ffn, err = el.Norm2.Forward(residual1); if err != nil { return nil, fmt.Errorf("enc Norm2: %w", err) }
	if el.IsMoE {
		if el.MoELayer==nil{return nil,fmt.Errorf("enc MoELayer nil")}; sublayer_output,err = el.MoELayer.Forward(norm_output_for_ffn,isTraining); if err!=nil{return nil,fmt.Errorf("enc MoE fwd: %w",err)}
	} else {
		if el.FeedForward==nil{return nil,fmt.Errorf("enc FFN nil")}; sublayer_output,err = el.FeedForward.Forward(norm_output_for_ffn,isTraining); if err!=nil{return nil,fmt.Errorf("enc FFN fwd: %w",err)}
	}
	sublayer_output, err = el.Dropout.Forward(sublayer_output, isTraining); if err != nil { return nil, fmt.Errorf("enc FFNDropout: %w", err) }
	finalResidual, err = Add(residual1, sublayer_output); if err != nil { return nil, fmt.Errorf("enc Res2: %w", err) }
	return finalResidual, nil
}

func (dl *DecoderLayerWithTensors) Forward(input *Tensor, encoderOutput *Tensor, isTraining bool, srcMask, tgtMask *Tensor) (*Tensor, error) {
	var err error; var selfAttnOut,norm1Out,residual1,norm_output_for_cross_attn,crossAttnOut,residual2,norm_output_for_ffn,sublayer_output,finalResidual *Tensor
    norm1Out,err=dl.Norm1.Forward(input);if err!=nil{return nil,fmt.Errorf("dec Norm1: %w",err)}
    selfAttnOut,err=dl.SelfAttention.Forward(norm1Out,norm1Out,norm1Out,tgtMask,isTraining);if err!=nil{return nil,fmt.Errorf("dec SelfAttn: %w",err)}
    selfAttnOut,err=dl.Dropout.Forward(selfAttnOut,isTraining);if err!=nil{return nil,fmt.Errorf("dec SelfAttnDrop: %w",err)}
    residual1,err=Add(input,selfAttnOut);if err!=nil{return nil,fmt.Errorf("dec Res1: %w",err)}
    norm_output_for_cross_attn,err=dl.Norm2.Forward(residual1);if err!=nil{return nil,fmt.Errorf("dec Norm2: %w",err)}
    crossAttnOut,err=dl.CrossAttention.Forward(norm_output_for_cross_attn,encoderOutput,encoderOutput,srcMask,isTraining);if err!=nil{return nil,fmt.Errorf("dec CrossAttn: %w",err)}
	crossAttnOut,err=dl.Dropout.Forward(crossAttnOut,isTraining);if err!=nil{return nil,fmt.Errorf("dec CrossAttnDrop: %w",err)}
    residual2,err=Add(residual1,crossAttnOut);if err!=nil{return nil,fmt.Errorf("dec Res2: %w",err)}
    norm_output_for_ffn,err=dl.Norm3.Forward(residual2);if err!=nil{return nil,fmt.Errorf("dec Norm3: %w",err)}
    if dl.IsMoE{
		if dl.MoELayer==nil{return nil,fmt.Errorf("dec MoELayer nil")};sublayer_output,err=dl.MoELayer.Forward(norm_output_for_ffn,isTraining);if err!=nil{return nil,fmt.Errorf("dec MoE fwd: %w",err)}
    }else{
		if dl.FeedForward==nil{return nil,fmt.Errorf("dec FFN nil")};sublayer_output,err=dl.FeedForward.Forward(norm_output_for_ffn,isTraining);if err!=nil{return nil,fmt.Errorf("dec FFN fwd: %w",err)}
    }
	sublayer_output,err=dl.Dropout.Forward(sublayer_output,isTraining);if err!=nil{return nil,fmt.Errorf("dec FFNDropout: %w",err)}
    finalResidual,err=Add(residual2,sublayer_output);if err!=nil{return nil,fmt.Errorf("dec Res3: %w",err)}
    return finalResidual,nil
}

// Forward methods for sub-components (MHA, FFN, LN) made graph-aware
func (mha *MultiHeadAttentionWithTensors) Forward(query, key, value *Tensor, mask *Tensor, isTraining bool) (*Tensor, error) {
	var qFull, kFull, vFull, err = MatMul(query,mha.QueryWeight); if err!=nil{return nil,fmt.Errorf("mha query proj: %w",err)}; kFull,err=MatMul(key,mha.KeyWeight); if err!=nil{return nil,fmt.Errorf("mha key proj: %w",err)}; vFull,err=MatMul(value,mha.ValueWeight); if err!=nil{return nil,fmt.Errorf("mha value proj: %w",err)}
	hOutputs:=make([]*Tensor,0,mha.NumHeads); var opErr error
	for h:=0;h<mha.NumHeads;h++{
		sC:=h*mha.HeadDim; pfx:=fmt.Sprintf("%s_h%d",query.Name,h)
		qH,e:=SliceColsTensor(qFull,sC,mha.HeadDim,pfx+"_q");if e!=nil{opErr=e;break}; kH,e:=SliceColsTensor(kFull,sC,mha.HeadDim,pfx+"_k");if e!=nil{opErr=e;break}; vH,e:=SliceColsTensor(vFull,sC,mha.HeadDim,pfx+"_v");if e!=nil{opErr=e;break}
		kT,e:=TensorTranspose(kH);if e!=nil{opErr=e;break}; scr,e:=MatMul(qH,kT);if e!=nil{opErr=e;break}
		scldScr,e:=ScalarMultiply(scr,1.0/math.Sqrt(float64(mha.HeadDim)));if e!=nil{opErr=e;break}
		mskScr:=scldScr;if mask!=nil{mskScr,e=ApplyAttentionMaskTensor(scldScr,mask,-1e9,pfx+"_mskScr");if e!=nil{opErr=e;break}}
		attW,e:=TensorSoftmax(mskScr,-1);if e!=nil{opErr=e;break}
		if mha.AttentionDropout!=nil{attW,e=mha.AttentionDropout.Forward(attW,isTraining);if e!=nil{opErr=e;break}}
		wVal,e:=MatMul(attW,vH);if e!=nil{opErr=e;break};hOutputs=append(hOutputs,wVal)
	}; if opErr!=nil{return nil,opErr}; if len(hOutputs)==0&&mha.NumHeads>0{return nil,fmt.Errorf("no head outputs")}; if mha.NumHeads==0{return nil,fmt.Errorf("0 heads")}
	var catOut *Tensor; if len(hOutputs)==1{catOut=hOutputs[0]}else{catOut,err=ConcatenateColsTensor(hOutputs,query.Name+"_cat_heads");if err!=nil{return nil,err}}
	return MatMul(catOut,mha.OutputWeight)
}

func (ff *FeedForwardWithTensors) Forward(input *Tensor, isTraining bool) (*Tensor, error) {
	var h, hDrop, output *Tensor; var err error
	h, err = MatMul(input, ff.W1); if err != nil { return nil, fmt.Errorf("ffn w1 mul: %w", err)}
	h, err = Add(h, ff.B1); if err != nil { return nil, fmt.Errorf("ffn b1 add: %w", err)}
	if ff.Activation==nil{return nil,fmt.Errorf("FFN activation nil")}; h,err=ff.Activation(h); if err!=nil{return nil, fmt.Errorf("ffn activation: %w",err)}
	hDrop,err=ff.Dropout.Forward(h,isTraining); if err!=nil{return nil, fmt.Errorf("ffn dropout: %w",err)}
	output, err = MatMul(hDrop, ff.W2); if err != nil { return nil, fmt.Errorf("ffn w2 mul: %w", err)}
	output, err = Add(output, ff.B2); if err != nil { return nil, fmt.Errorf("ffn b2 add: %w", err)}
	return output, nil
}

func (ln *LayerNormWithTensors) Forward(input *Tensor) (*Tensor, error) {
	epsT := NewScalarTensor(ln.Eps, input.Graph, false)

	meanVal,e:=TensorMean(input,1,true); if e!=nil{return nil,fmt.Errorf("ln mean: %w",e)}
	cenIn,e:=Subtract(input,meanVal); if e!=nil{return nil,fmt.Errorf("ln center: %w",e)}
	sqIn,e:=TensorSquare(cenIn); if e!=nil{return nil,fmt.Errorf("ln square: %w",e)}
	varV,e:=TensorMean(sqIn,1,true); if e!=nil{return nil,fmt.Errorf("ln variance: %w",e)}
	stdVTemp,e:=Add(varV,epsT); if e!=nil{return nil,fmt.Errorf("ln add_eps: %w",e)}
	stdV,e:=TensorSqrt(stdVTemp); if e!=nil{return nil,fmt.Errorf("ln sqrt: %w",e)}
	normIn,e:=TensorDivide(cenIn,stdV); if e!=nil{return nil,fmt.Errorf("ln divide: %w",e)}
	
	scaled,e:=Multiply(normIn,ln.Gamma); if e!=nil{return nil,fmt.Errorf("ln scale: %w",e)}
	return Add(scaled,ln.Beta)
}
