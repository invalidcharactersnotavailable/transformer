package autodiff

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// DataType (renamed from ScalarType for clarity with Tensor's DType field)
type DataType string
const (
	Float64 DataType = "float64" // Default
	Int64   DataType = "int64"   // Represents integer-like values, stored as float64
	// Future: Float32, Int32, Bool etc.
)

// Tensor structure
type Tensor struct {
	Data         *Matrix
	Grad         *Matrix
	Graph        *ComputationGraph
	RequiresGrad bool
	BackwardFn   func()
	Children     []*Tensor
	Name         string
	DType        DataType // Use DType from TensorConfig
	shape        []int    // For >2D, current ops assume 2D via Data.Rows/Cols
}
type ComputationGraph struct { nodes []*Tensor }
func NewComputationGraph() *ComputationGraph { return &ComputationGraph{nodes: make([]*Tensor, 0)} }
func (g *ComputationGraph) addNode(t *Tensor) {
	if t != nil && t.RequiresGrad && t.BackwardFn != nil {
		isNewNode := true; for _, n := range g.nodes { if n == t { isNewNode = false; break } }; if isNewNode { g.nodes = append(g.nodes, t) }
	}
}
func (g *ComputationGraph) Backward() {
	for i := len(g.nodes) - 1; i >= 0; i-- { if g.nodes[i].BackwardFn != nil { g.nodes[i].BackwardFn() } }
	g.nodes = make([]*Tensor, 0)
}
type TensorConfig struct { RequiresGrad bool; Name string; Graph *ComputationGraph; DType DataType }
func (tc *TensorConfig) GraphOpt() *ComputationGraph { if tc == nil { return nil }; return tc.Graph }
func DefaultTensorConfig() *TensorConfig { return &TensorConfig{RequiresGrad: false, Name: "", Graph: nil, DType: Float64} }

func NewTensor(data *Matrix, config *TensorConfig) (*Tensor, error) {
	if data == nil { return nil, fmt.Errorf("data matrix cannot be nil") }
	cfg := config; if cfg == nil { cfg = DefaultTensorConfig() }
	if cfg.DType == "" { cfg.DType = Float64 } // Default DType if not specified
	var grad *Matrix; var err error
	if cfg.RequiresGrad { grad, err = NewMatrix(data.Rows, data.Cols); if err != nil { return nil, fmt.Errorf("create grad: %w", err) } }
	return &Tensor{ Data: data, Grad: grad, Graph: cfg.Graph, RequiresGrad: cfg.RequiresGrad, Name: cfg.Name, DType: cfg.DType, shape: []int{data.Rows, data.Cols} }, nil
}

func NewScalarTensor(value float64, graph *ComputationGraph, requiresGrad ...bool) *Tensor {
    reqGrad := false
    if len(requiresGrad) > 0 { reqGrad = requiresGrad[0] }
    data, _ := NewMatrix(1,1); data.Data[0][0] = value
    name := fmt.Sprintf("scalar_%.2f",value); if reqGrad { name += "_rg"}
    t, _ := NewTensor(data, &TensorConfig{Graph: graph, RequiresGrad: reqGrad, Name: name, DType: Float64})
    return t
}
func (t *Tensor) Shape() []int { if t.Data == nil { return []int{0,0} }; return []int{t.Data.Rows, t.Data.Cols} }
func (t *Tensor) SetGraph(g *ComputationGraph) { t.Graph = g }
func (t *Tensor) SetRequiresGrad(requires bool) {
    t.RequiresGrad = requires
    if requires && t.Grad == nil && t.Data != nil { grad, _ := NewMatrix(t.Data.Rows, t.Data.Cols); t.Grad = grad
    } else if !requires { t.Grad = nil }
}
func NewRandomTensor(rows, cols int, config *TensorConfig) (*Tensor, error) {
	if rows < 0 || cols < 0 { return nil, fmt.Errorf("dims must be non-negative") }
	cfg := config; if cfg == nil { cfg = DefaultTensorConfig() }
	data, err := NewRandomMatrix(rows, cols); if err != nil { return nil, err }
	return NewTensor(data, cfg)
}
func NewUniformTensor(rows, cols int, min, max float64, config *TensorConfig) (*Tensor, error) {
    if rows < 0 || cols < 0 { return nil, fmt.Errorf("dims must be non-negative") }
    cfg := config; if cfg == nil { cfg = DefaultTensorConfig() }
    data, err := NewUniformMatrix(rows, cols, min, max); if err != nil { return nil, err }
    return NewTensor(data, cfg)
}
func NewZerosTensor(cfgOrNil interface{}, shape ...int) (*Tensor, error) {
    var cfg *TensorConfig; actualShape := shape
    if len(shape) > 0 {
		if c, ok := shape[len(shape)-1].(*TensorConfig); ok { cfg = c; actualShape = shape[:len(shape)-1]
		} else if c, ok := cfgOrNil.(*TensorConfig); ok { cfg = c }
    }
	if cfg == nil && cfgOrNil != nil { if c, ok := cfgOrNil.(*TensorConfig); ok { cfg = c } }
    if cfg == nil { cfg = DefaultTensorConfig() }

    if len(actualShape) == 0 { return nil, fmt.Errorf("shape cannot be empty for NewZerosTensor") }
    rows := actualShape[0]; cols := 1
    if len(actualShape) > 1 { cols = actualShape[1] }; if len(actualShape) > 2 { return nil, fmt.Errorf("only 2D") }
    if rows < 0 || cols < 0 { return nil, fmt.Errorf("dims non-negative") };

    data,err:=NewMatrix(rows,cols); if err!=nil{return nil,err}; return NewTensor(data,cfg)
}
func NewTensorFromData(data [][]float64, config *TensorConfig) (*Tensor, error) {
	cfg := config; if cfg == nil { cfg = DefaultTensorConfig() }
	rows := 0; cols := 0
	if len(data) > 0 { rows = len(data); cols = len(data[0]) }
	mat:=&Matrix{Rows:rows,Cols:cols,Data:data}; return NewTensor(mat,cfg)
}
func (t *Tensor) ZeroGrad() {
	if !t.RequiresGrad||t.Grad==nil{return}; for i:=0;i<t.Grad.Rows;i++{for j:=0;j<t.Grad.Cols;j++{t.Grad.Data[i][j]=0.0}}
}
func (t *Tensor) BackwardAll() error {
	if t.Graph==nil && t.RequiresGrad { t.Graph = NewComputationGraph(); if t.BackwardFn != nil {t.Graph.addNode(t)} } else if t.Graph == nil {return fmt.Errorf("tensor %s not on graph for backward", t.Name)}
	if t.Grad==nil&&t.RequiresGrad{t.Grad,_=NewMatrix(t.Data.Rows,t.Data.Cols)}
	if t.Grad!=nil{if t.Data.Rows==1&&t.Data.Cols==1{t.Grad.Data[0][0]=1.0}else{for i:=0;i<t.Grad.Rows;i++{for j:=0;j<t.Grad.Cols;j++{t.Grad.Data[i][j]=1.0}}}}
	t.Graph.Backward(); return nil
}

// Core Ops
func MatMul(a,b *Tensor) (*Tensor,error) {
	if a==nil||b==nil{return nil,fmt.Errorf("inputs nil")};if a.Data.Cols!=b.Data.Rows{return nil,fmt.Errorf("dim mismatch")};g:=a.Graph;if g==nil{g=b.Graph}
	cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad||b.RequiresGrad,Name:fmt.Sprintf("MatMul(%s,%s)",a.Name,b.Name),Graph:g}
	dat,err:=matrixMatMul(a.Data,b.Data);if err!=nil{return nil,err};res,err:=NewTensor(dat,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a,b);res.BackwardFn=func(){
		if a.RequiresGrad{if a.Grad==nil{a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)};bT,_:=Transpose(b.Data);dLdA,_:=matrixMatMul(res.Grad,bT);for i:=0;i<a.Grad.Rows;i++{for j:=0;j<a.Grad.Cols;j++{a.Grad.Data[i][j]+=dLdA.Data[i][j]}}}
		if b.RequiresGrad{if b.Grad==nil{b.Grad,_=NewMatrix(b.Data.Rows,b.Data.Cols)};aT,_:=Transpose(a.Data);dLdB,_:=matrixMatMul(aT,res.Grad);for i:=0;i<b.Grad.Rows;i++{for j:=0;j<b.Grad.Cols;j++{b.Grad.Data[i][j]+=dLdB.Data[i][j]}}}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func Add(a,b *Tensor) (*Tensor,error) {
	if a==nil||b==nil{return nil,fmt.Errorf("inputs nil")};outR,outC,ok:=checkBroadcastShapes(a,b);if !ok{return nil,fmt.Errorf("shape mismatch Add/Broadcast: a(%v) b(%v)", a.Shape(), b.Shape())}
	g:=a.Graph;if g==nil{g=b.Graph};cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad||b.RequiresGrad,Name:fmt.Sprintf("Add(%s,%s)",a.Name,b.Name),Graph:g}
	dat,_:=NewMatrix(outR,outC);for i:=0;i<outR;i++{for j:=0;j<outC;j++{dat.Data[i][j]=a.Data.Data[i%a.Data.Rows][j%a.Data.Cols]+b.Data.Data[i%b.Data.Rows][j%b.Data.Cols]}}
	res,err:=NewTensor(dat,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a,b);res.BackwardFn=func(){
		if a.RequiresGrad{if a.Grad==nil{a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)};for i:=0;i<outR;i++{for j:=0;j<outC;j++{a.Grad.Data[i%a.Data.Rows][j%a.Data.Cols]+=res.Grad.Data[i][j]}}}
		if b.RequiresGrad{if b.Grad==nil{b.Grad,_=NewMatrix(b.Data.Rows,b.Data.Cols)};for i:=0;i<outR;i++{for j:=0;j<outC;j++{b.Grad.Data[i%b.Data.Rows][j%b.Data.Cols]+=res.Grad.Data[i][j]}}}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func Multiply(a,b *Tensor) (*Tensor,error) {
	if a==nil||b==nil{return nil,fmt.Errorf("inputs nil")};outR,outC,ok:=checkBroadcastShapes(a,b);if !ok{return nil,fmt.Errorf("shape mismatch Multiply/Broadcast: a(%v) b(%v)", a.Shape(), b.Shape())}
	g:=a.Graph;if g==nil{g=b.Graph};cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad||b.RequiresGrad,Name:fmt.Sprintf("Multiply(%s,%s)",a.Name,b.Name),Graph:g}
	dat,_:=NewMatrix(outR,outC);for i:=0;i<outR;i++{for j:=0;j<outC;j++{dat.Data[i][j]=a.Data.Data[i%a.Data.Rows][j%a.Data.Cols]*b.Data.Data[i%b.Data.Rows][j%b.Data.Cols]}}
	res,err:=NewTensor(dat,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a,b);res.BackwardFn=func(){
		if a.RequiresGrad{if a.Grad==nil{a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)};for i:=0;i<outR;i++{for j:=0;j<outC;j++{a.Grad.Data[i%a.Data.Rows][j%a.Data.Cols]+=res.Grad.Data[i][j]*b.Data.Data[i%b.Data.Rows][j%b.Data.Cols]}}}
		if b.RequiresGrad{if b.Grad==nil{b.Grad,_=NewMatrix(b.Data.Rows,b.Data.Cols)};for i:=0;i<outR;i++{for j:=0;j<outC;j++{b.Grad.Data[i%b.Data.Rows][j%b.Data.Cols]+=res.Grad.Data[i][j]*a.Data.Data[i%a.Data.Rows][j%a.Data.Cols]}}}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func checkBroadcastShapes(a,b *Tensor) (int,int,bool) {
	aR,aC,bR,bC := a.Data.Rows,a.Data.Cols,b.Data.Rows,b.Data.Cols; outR,outC := aR,aC
	if(aR==bR&&aC==bC)|| (aR==bR&&bC==1&&aC>=1)|| (aC==bC&&bR==1&&aR>=1)|| (bR==aR&&aC==1&&bC>=1)|| (bC==aC&&aR==1&&bR>=1){
		if bR>outR{outR=bR};if bC>outC{outC=bC}; return outR,outC,true
	}; return 0,0,false
}
func ScalarMultiply(a *Tensor, scalar float64) (*Tensor, error) {
	if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for ScalarMultiply") }
	g := a.Graph; cfg := &TensorConfig{ RequiresGrad: a.RequiresGrad, Name: fmt.Sprintf("ScalarMultiply(%s,%.2f)",a.Name, scalar), Graph: g }
	resD, _ := NewMatrix(a.Data.Rows, a.Data.Cols); for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{resD.Data[i][j]=a.Data.Data[i][j]*scalar}}
	res,err:=NewTensor(resD,cfg); if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.RequiresGrad{if a.Grad==nil{a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)};for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{a.Grad.Data[i][j]+=res.Grad.Data[i][j]*scalar}}}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func GELU(a *Tensor) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil for GELU")};g:=a.Graph;cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:fmt.Sprintf("GELU(%s)",a.Name),Graph:g}
	resD,_:=NewMatrix(a.Data.Rows,a.Data.Cols);sqrt2Pi:=math.Sqrt(2.0/math.Pi);coeff:=0.044715
	for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{x:=a.Data.Data[i][j];tanhArg:=sqrt2Pi*(x+coeff*math.Pow(x,3.0));resD.Data[i][j]=0.5*x*(1.0+math.Tanh(tanhArg))}}
	res,err:=NewTensor(resD,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
		if a.Grad != nil && res.Grad != nil {
			for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{x:=a.Data.Data[i][j];tanhArg:=sqrt2Pi*(x+coeff*math.Pow(x,3.0));tanhV:=math.Tanh(tanhArg);dTanh:=1.0-tanhV*tanhV;inDeriv:=sqrt2Pi*(1.0+3.0*coeff*math.Pow(x,2.0));geluGrad:=0.5*(1.0+tanhV)+0.5*x*dTanh*inDeriv;a.Grad.Data[i][j]+=res.Grad.Data[i][j]*geluGrad}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func TensorSoftmax(a *Tensor, axis int) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil for Softmax")};if axis!=1&&axis!=-1{return nil,fmt.Errorf("Softmax only axis=1/-1")};g:=a.Graph
	cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:fmt.Sprintf("Softmax(%s)",a.Name),Graph:g}
	resD,_:=matrixSoftmax(a.Data);res,err:=NewTensor(resD,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
		if a.Grad != nil && res.Grad != nil {
			for r:=0;r<res.Grad.Rows;r++{for c:=0;c<res.Grad.Cols;c++{s_rc:=res.Data.Data[r][c];dL_ds_rc:=res.Grad.Data[r][c];rowSumDotGrad:=0.0;for k:=0;k<res.Data.Cols;k++{rowSumDotGrad+=res.Grad.Data[r][k]*res.Data.Data[r][k]};a.Grad.Data[r][c]+=s_rc*(dL_ds_rc-rowSumDotGrad)}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func TensorSquare(a *Tensor) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil for Square")};g:=a.Graph;cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:a.Name+"_squared",Graph:g}
	resD,_:=NewMatrix(a.Data.Rows,a.Data.Cols);for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{resD.Data[i][j]=a.Data.Data[i][j]*a.Data.Data[i][j]}}
	res,err:=NewTensor(resD,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
		if a.Grad != nil && res.Grad != nil {
			for i:=0;i<a.Grad.Rows;i++{for j:=0;j<a.Grad.Cols;j++{a.Grad.Data[i][j]+=2*a.Data.Data[i][j]*res.Grad.Data[i][j]}}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func TensorMean(a *Tensor, axis int, keepDims bool) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil for Mean")};g:=a.Graph;outR,outC:=0,0;var totalElemPerReduction float64
	inputRows, inputCols := a.Shape()[0], a.Shape()[1]
	if axis==-1{outR,outC=1,1;totalElemPerReduction=float64(inputRows*inputCols)} else if axis==0{outR=1;if keepDims{outR=1};outC=inputCols;totalElemPerReduction=float64(inputRows)} else if axis==1{outR=inputRows;outC=1;if keepDims{outC=1};totalElemPerReduction=float64(inputCols)} else{return nil,fmt.Errorf("unsupported axis for Mean:%d",axis)}
	if totalElemPerReduction==0 && !(inputRows==0 || inputCols==0) {
		return NewZerosTensor(&TensorConfig{Graph:g,Name:a.Name+"_mean_empty_reduction"}, outR, outC)
	}

	cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:a.Name+"_mean",Graph:g};resData,_:=NewMatrix(outR,outC)
	if totalElemPerReduction == 0 {
	} else if axis==-1{sum:=0.0;for i:=0;i<inputRows;i++{for j:=0;j<inputCols;j++{sum+=a.Data.Data[i][j]}};resData.Data[0][0]=sum/totalElemPerReduction} else if axis==0{for j:=0;j<inputCols;j++{sum:=0.0;for i:=0;i<inputRows;i++{sum+=a.Data.Data[i][j]};resData.Data[0][j]=sum/totalElemPerReduction}} else{/*axis==1*/for i:=0;i<inputRows;i++{sum:=0.0;for j:=0;j<inputCols;j++{sum+=a.Data.Data[i][j]};resData.Data[i][0]=sum/totalElemPerReduction}}
	res,err:=NewTensor(resData,cfg);if err!=nil{return nil,err}

	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(inputRows,inputCols)};
		if a.Grad != nil && res.Grad != nil && totalElemPerReduction != 0 {
			gradPortion:=1.0/totalElemPerReduction
			if axis==-1{gr:=res.Grad.Data[0][0]*gradPortion;for i:=0;i<inputRows;i++{for j:=0;j<inputCols;j++{a.Grad.Data[i][j]+=gr}}} else if axis==0{for j:=0;j<outC;j++{gr:=res.Grad.Data[0][j]*gradPortion;for i:=0;i<inputRows;i++{a.Grad.Data[i][j]+=gr}}} else{/*axis==1*/for i:=0;i<outR;i++{gr:=res.Grad.Data[i][0]*gradPortion;for j:=0;j<inputCols;j++{a.Grad.Data[i][j]+=gr}}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func TensorLogSumExp(a *Tensor, axis int, keepDims bool) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil for LogSumExp")};g:=a.Graph;var outR,outC int
	inputRows, inputCols := a.Shape()[0], a.Shape()[1]
	if axis==-1{outR,outC=1,1} else if axis==0{outR=1;if keepDims{outR=1};outC=inputCols} else if axis==1{outR=inputRows;outC=1;if keepDims{outC=1}} else{return nil,fmt.Errorf("unsupported axis for LogSumExp:%d",axis)}
	if (axis == -1 && inputRows*inputCols == 0) || (axis == 0 && inputRows == 0) || (axis == 1 && inputCols == 0) {
		fmt.Printf("Warning: LogSumExp over zero elements for axis %d. Returning zero tensor.\n", axis)
		return NewZerosTensor(&TensorConfig{Graph:g, Name:a.Name+"_logsumexp_empty"}, outR, outC)
	}

	cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:a.Name+"_logsumexp",Graph:g};resData,_:=NewMatrix(outR,outC);softmaxedData,_:=NewMatrix(inputRows,inputCols)
	if axis==-1{maxV:=-math.MaxFloat64;for i:=0;i<inputRows;i++{for j:=0;j<inputCols;j++{if a.Data.Data[i][j]>maxV{maxV=a.Data.Data[i][j]}}};sumExp:=0.0;for i:=0;i<inputRows;i++{for j:=0;j<inputCols;j++{v:=math.Exp(a.Data.Data[i][j]-maxV);softmaxedData.Data[i][j]=v;sumExp+=v}};if sumExp==0{resData.Data[0][0]=-math.MaxFloat64}else{resData.Data[0][0]=maxV+math.Log(sumExp)};if sumExp!=0{for i:=0;i<inputRows;i++{for j:=0;j<inputCols;j++{softmaxedData.Data[i][j]/=sumExp}}}} else if axis==0{for j:=0;j<inputCols;j++{maxV:=-math.MaxFloat64;for i:=0;i<inputRows;i++{if a.Data.Data[i][j]>maxV{maxV=a.Data.Data[i][j]}};sumExp:=0.0;for i:=0;i<inputRows;i++{v:=math.Exp(a.Data.Data[i][j]-maxV);softmaxedData.Data[i][j]=v;sumExp+=v};if sumExp==0{resData.Data[0][j]=-math.MaxFloat64}else{resData.Data[0][j]=maxV+math.Log(sumExp)};if sumExp!=0{for i:=0;i<inputRows;i++{softmaxedData.Data[i][j]/=sumExp}}}} else{/*axis==1*/for i:=0;i<inputRows;i++{maxV:=-math.MaxFloat64;for j:=0;j<inputCols;j++{if a.Data.Data[i][j]>maxV{maxV=a.Data.Data[i][j]}};sumExp:=0.0;for j:=0;j<inputCols;j++{v:=math.Exp(a.Data.Data[i][j]-maxV);softmaxedData.Data[i][j]=v;sumExp+=v};if sumExp==0{resData.Data[i][0]=-math.MaxFloat64}else{resData.Data[i][0]=maxV+math.Log(sumExp)};if sumExp!=0{for j:=0;j<inputCols;j++{softmaxedData.Data[i][j]/=sumExp}}}}
	res,err:=NewTensor(resData,cfg);if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(inputRows,inputCols)}
		if a.Grad != nil && res.Grad != nil {
			if axis==-1{gr:=res.Grad.Data[0][0];for i:=0;i<inputRows;i++{for j:=0;j<inputCols;j++{a.Grad.Data[i][j]+=softmaxedData.Data[i][j]*gr}}} else if axis==0{for j:=0;j<outC;j++{gr:=res.Grad.Data[0][j];for i:=0;i<inputRows;i++{a.Grad.Data[i][j]+=softmaxedData.Data[i][j]*gr}}} else{/*axis==1*/for i:=0;i<outR;i++{gr:=res.Grad.Data[i][0];for j:=0;j<inputCols;j++{a.Grad.Data[i][j]+=softmaxedData.Data[i][j]*gr}}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}

// SliceArg structure
type SliceArg struct { Start, End int }
// TensorSlice implementation
func TensorSlice(a *Tensor, args []*SliceArg, nameSuffix ...string) (*Tensor, error) {
	if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for Slice") }
	if len(args) != 2 { return nil, fmt.Errorf("TensorSlice currently expects 2 slice arguments for 2D tensor") }
	graph := a.Graph
	var name string
	if len(nameSuffix)>0 && nameSuffix[0]!="" {name = nameSuffix[0]} else {name = fmt.Sprintf("Slice(%s)", a.Name)}

	cfg := &TensorConfig{ RequiresGrad: a.RequiresGrad, Name: name, Graph: graph, DType: a.dtype }
	rowArg, colArg := args[0], args[1]
	rStart, rEnd, cStart, cEnd := rowArg.Start, rowArg.End, colArg.Start, colArg.End

	if rStart < 0 { rStart = 0 }; if rEnd > a.Data.Rows { rEnd = a.Data.Rows }
	if cStart < 0 { cStart = 0 }; if cEnd > a.Data.Cols { cEnd = a.Data.Cols }
	if rStart > rEnd  { rStart = rEnd }
	if cStart > cEnd { cStart = cEnd }

	outRows, outCols := rEnd - rStart, cEnd - cStart
		
	resD, _ := NewMatrix(outRows, outCols);
	for i:=0;i<outRows;i++{for j:=0;j<outCols;j++{
		if (rStart+i) < a.Data.Rows && (cStart+j) < a.Data.Cols {
			resD.Data[i][j]=a.Data.Data[rStart+i][cStart+j]
		}
	}}
	res,err:=NewTensor(resD,cfg); if err!=nil{return nil,err}
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
		if a.Grad != nil && res.Grad != nil {
			for i:=0;i<outRows;i++{for j:=0;j<outCols;j++{
				if (rStart+i) < a.Grad.Rows && (cStart+j) < a.Grad.Cols {
					a.Grad.Data[rStart+i][cStart+j]+=res.Grad.Data[i][j]
				}
			}}
		}
	};if graph!=nil{graph.addNode(res)}}; return res,nil
}

func TensorSqrt(a *Tensor) (*Tensor, error) {
    if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for Sqrt") }
    graph := a.Graph
    cfg := &TensorConfig{RequiresGrad: a.RequiresGrad, Name: fmt.Sprintf("Sqrt(%s)", a.Name), Graph: graph}
    resD, _ := NewMatrix(a.Data.Rows, a.Data.Cols)
    for i := 0; i < a.Data.Rows; i++ { for j := 0; j < a.Data.Cols; j++ {
		if a.Data.Data[i][j] < 0 { return nil, fmt.Errorf("cannot compute sqrt of negative number: %f at [%d,%d]", a.Data.Data[i][j],i,j)}
		resD.Data[i][j] = math.Sqrt(a.Data.Data[i][j])
    } }
    res, err := NewTensor(resD, cfg); if err != nil { return nil, err }
    if res.RequiresGrad {
        res.Children = append(res.Children, a)
        res.BackwardFn = func() {
            if a.Grad == nil && a.RequiresGrad { a.Grad, _ = NewMatrix(a.Data.Rows, a.Data.Cols) }
			if a.Grad != nil && res.Grad != nil {
				for i := 0; i < a.Data.Rows; i++ { for j := 0; j < a.Data.Cols; j++ {
                    denominator := res.Data.Data[i][j] + 1e-12
                    a.Grad.Data[i][j] += res.Grad.Data[i][j] * 0.5 / denominator
				} }
			}
        }
        if graph != nil { graph.addNode(res) }
    }
    return res, nil
}
func TensorDivide(a, b *Tensor) (*Tensor, error) {
    if a==nil||b==nil{return nil,fmt.Errorf("inputs nil for Divide")};outR,outC,ok:=checkBroadcastShapes(a,b);if !ok{return nil,fmt.Errorf("shape mismatch Divide/Broadcast: a(%v) b(%v)",a.Shape(),b.Shape())}
	g:=a.Graph;if g==nil{g=b.Graph};cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad||b.RequiresGrad,Name:fmt.Sprintf("Divide(%s,%s)",a.Name,b.Name),Graph:g}
    resD,_:=NewMatrix(outR,outC);for i:=0;i<outR;i++{for j:=0;j<outC;j++{
        valB := b.Data.Data[i%b.Data.Rows][j%b.Data.Cols]
        if math.Abs(valB) < 1e-12 { return nil, fmt.Errorf("division by near-zero value at b[%d,%d]", i%b.Data.Rows, j%b.Data.Cols)}
        resD.Data[i][j] = a.Data.Data[i%a.Data.Rows][j%a.Data.Cols] / valB
    } }
    res,err:=NewTensor(resD,cfg);if err!=nil{return nil,err}
    if res.RequiresGrad{res.Children=append(res.Children,a,b);res.BackwardFn=func(){
        if a.RequiresGrad{
            if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
			if a.Grad != nil && res.Grad != nil {
				for i:=0;i<outR;i++{for j:=0;j<outC;j++{
					valB := b.Data.Data[i%b.Data.Rows][j%b.Data.Cols]; if math.Abs(valB) < 1e-12 { continue }; a.Grad.Data[i%a.Data.Rows][j%a.Data.Cols]+=res.Grad.Data[i][j]/valB
				}}
			}
        }
        if b.RequiresGrad{
            if b.Grad==nil && b.RequiresGrad {b.Grad,_=NewMatrix(b.Data.Rows,b.Data.Cols)}
			if b.Grad != nil && res.Grad != nil {
				for i:=0;i<outR;i++{for j:=0;j<outC;j++{
					valA := a.Data.Data[i%a.Data.Rows][j%a.Data.Cols]; valB := b.Data.Data[i%b.Data.Rows][j%b.Data.Cols]; if math.Abs(valB) < 1e-12 { continue }; b.Grad.Data[i%b.Data.Rows][j%b.Data.Cols]+=res.Grad.Data[i][j]*(-valA/(valB*valB))
				}}
			}
        }
    };if g!=nil{g.addNode(res)}};return res,nil
}

// Placeholders & More Functional Ops
func TensorUnsqueeze(a *Tensor, axis int) (*Tensor, error) {
	if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for Unsqueeze") }
	graph := a.Graph
	cloned, err := a.Clone(); if err != nil { return nil, err }
	cloned.Name = a.Name + "_unsqueezed"; cloned.Graph = graph
	// Actual shape change logic is missing due to 2D Matrix backend.
	if cloned.RequiresGrad {
		cloned.Children = append(cloned.Children, a)
		cloned.BackwardFn = func() {
			if a.Grad == nil && a.RequiresGrad {a.Grad, _ = NewMatrix(a.Data.Rows, a.Data.Cols)}
			if a.Grad != nil && cloned.Grad != nil {
				for i := 0; i < cloned.Grad.Rows; i++ { for j := 0; j < cloned.Grad.Cols; j++ {
					if i < a.Grad.Rows && j < a.Grad.Cols { a.Grad.Data[i][j] += cloned.Grad.Data[i][j] }
				} }
			}
		}
		if graph != nil && cloned.RequiresGrad {graph.addNode(cloned)}
	}
	return cloned, nil
}
func TensorCast(a *Tensor, newType DataType) (*Tensor, error) {
	if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for Cast") }
	graph := a.Graph
	// fmt.Printf("Warning: TensorCast is conceptual; current backend is float64. Casting to %s.\n", newType)
	resultData, _ := a.Data.Clone()
	// If casting to an integer type, truncate the data.
	if newType == Int64 || newType == Int32 { // Conceptual int types
		fmt.Printf("TensorCast to %s: truncating float64 data.\n", newType)
		for i := 0; i < resultData.Rows; i++ {
			for j := 0; j < resultData.Cols; j++ {
				resultData.Data[i][j] = math.Trunc(resultData.Data[i][j])
			}
		}
	} else if newType != Float64 {
		fmt.Printf("Warning: TensorCast to %s is conceptual; current backend is float64. No data change other than potential truncation for int types.\n", newType)
	}

	config := &TensorConfig{ RequiresGrad: a.RequiresGrad, Name: a.Name + "_cast_to_" + string(newType), Graph: graph, DType: newType}
	result, err := NewTensor(resultData, config); if err != nil { return nil, err }

	if result.RequiresGrad {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() {
			if a.Grad == nil && a.RequiresGrad {a.Grad, _ = NewMatrix(a.Data.Rows, a.Data.Cols)}
			if a.Grad != nil && result.Grad != nil {
				// Gradient of cast: if float->float, it's 1.
				// If float->int (truncation), grad is 0 where fractional part was dropped, 1 otherwise (straight-through estimator idea, or just 0).
				// For simplicity, if original was float and target is int-like, pass grad as 0 for now.
				// If it's float to float (even if different precision conceptually), pass grad as 1.
				if a.DType == Float64 && (newType == Int32 || newType == Int64) {
					// Non-differentiable, so effectively zero gradient flows back to `a` due to this op.
					// However, `a` might receive gradients from other paths. We don't zero out a.Grad here.
					// The lack of `a.Grad.Data[i][j] += ...` means this op contributes zero to a's grad.
				} else {
					for i := 0; i < a.Grad.Rows; i++ { for j := 0; j < a.Grad.Cols; j++ { a.Grad.Data[i][j] += result.Grad.Data[i][j] } }
				}
			}
		}
		if graph != nil {graph.addNode(result)}
	}
	return result, nil
}
func TensorEqualScalar(a *Tensor, scalar float64) (*Tensor, error) {
	if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for EqualScalar") }
	graph := a.Graph
	resultData, _ := NewMatrix(a.Data.Rows, a.Data.Cols)
	for i := 0; i < a.Data.Rows; i++ { for j := 0; j < a.Data.Cols; j++ {
		if math.Abs(a.Data.Data[i][j]-scalar) < 1e-9 { resultData.Data[i][j] = 1.0 } else { resultData.Data[i][j] = 0.0 }
	} }
	config := &TensorConfig{ RequiresGrad: false, Name: a.Name + "_eq_scalar", Graph: graph, DType: Float64 }
	return NewTensor(resultData, config)
}

func TensorTopK(input *Tensor, k int, axis int, sorted bool) (values *Tensor, indices *Tensor, err error) {
	if input == nil { return nil, nil, fmt.Errorf("input tensor cannot be nil for TopK") }
	if k <= 0 { return nil, nil, fmt.Errorf("k must be positive") }
	if axis != 1 && axis != -1 { return nil, nil, fmt.Errorf("TensorTopK currently only supports axis=1 or -1 for 2D tensors, got %d", axis) }
	if input.Data.Cols == 0 && k > 0 { return nil, nil, fmt.Errorf("cannot select top K from tensor with 0 columns")}
	
	actualK := k
	if input.Data.Cols < k { actualK = input.Data.Cols }
	
	graph := input.Graph; numRows := input.Data.Rows
	
	topKValuesData := make([][]float64, numRows)
	topKIndicesData := make([][]float64, numRows)

	type pair struct { val float64; idx int }

	for i := 0; i < numRows; i++ {
		rowPairs := make([]pair, input.Data.Cols)
		for j := 0; j < input.Data.Cols; j++ { rowPairs[j] = pair{val: input.Data.Data[i][j], idx: j} }

		sort.SliceStable(rowPairs, func(p1, p2 int) bool { return rowPairs[p1].val > rowPairs[p2].val })

		topKValuesData[i] = make([]float64, actualK)
		topKIndicesData[i] = make([]float64, actualK)
		for j := 0; j < actualK; j++ {
			topKValuesData[i][j] = rowPairs[j].val
			topKIndicesData[i][j] = float64(rowPairs[j].idx)
		}
	}

	valuesConfig := &TensorConfig{ RequiresGrad: input.RequiresGrad, Name: input.Name + "_topk_values", Graph: graph }
	outValues, errV := NewTensorFromData(topKValuesData, valuesConfig); if errV != nil { return nil, nil, fmt.Errorf("error creating values tensor for TopK: %w", errV)}

	indicesConfig := &TensorConfig{ RequiresGrad: false, Name: input.Name + "_topk_indices", Graph: graph, DType: Int64 } // Indices are int-like
	outIndices, errI := NewTensorFromData(topKIndicesData, indicesConfig); if errI != nil { return nil, nil, fmt.Errorf("error creating indices tensor for TopK: %w", errI)}

	if input.RequiresGrad {
		outValues.Children=append(outValues.Children,input); outValues.BackwardFn=func(){
		if input.Grad==nil && input.RequiresGrad {input.Grad,_=NewMatrix(input.Data.Rows,input.Data.Cols)}
		if input.Grad != nil && outValues.Grad != nil {
			for i:=0;i<numRows;i++{for j:=0;j<actualK;j++{origIdx:=int(outIndices.Data.Data[i][j]);if i<input.Grad.Rows&&origIdx<input.Grad.Cols&&i<outValues.Grad.Rows&&j<outValues.Grad.Cols{input.Grad.Data[i][origIdx]+=outValues.Grad.Data[i][j]}}}}
		}
	};if graph!=nil && outValues.RequiresGrad {graph.addNode(outValues)}}
	return outValues,outIdx,nil
}
func TensorArgMax(input *Tensor, axis int, keepDims bool) (*Tensor, error) {
	if input == nil { return nil, fmt.Errorf("input tensor cannot be nil for ArgMax") }
	if axis != 1 && axis != -1 { return nil, fmt.Errorf("TensorArgMax currently only supports axis=1 or -1 for 2D") }
	if input.Data.Rows == 0 { return NewTensorFromData([][]float64{}, &TensorConfig{Graph: input.Graph, Name: input.Name + "_argmax", DType: Int64}) }
	if input.Data.Cols == 0 { return nil, fmt.Errorf("cannot compute ArgMax for tensor with 0 columns on axis 1")}
	numRows := input.Data.Rows; argMaxData := make([][]float64, numRows)
	for i := 0; i < numRows; i++ { argMaxData[i] = make([]float64, 1); maxVal := input.Data.Data[i][0]; maxIdx := 0
		for j := 1; j < input.Data.Cols; j++ { if input.Data.Data[i][j] > maxVal { maxVal = input.Data.Data[i][j]; maxIdx = j } }
		argMaxData[i][0] = float64(maxIdx)
	}
	cfg := &TensorConfig{RequiresGrad: false, Name: input.Name + "_argmax", Graph: input.Graph, DType: Int64}
	indices, err := NewTensorFromData(argMaxData, cfg); if err != nil { return nil, err }
	return indices, nil
}
func TensorMax(input *Tensor, axis int, keepDims bool) (*Tensor, error) {
	if input == nil { return nil, fmt.Errorf("input tensor cannot be nil for Max") }
	graph := input.Graph; var resData [][]float64; outR,outC,inR,inC := 0,0,input.Data.Rows,input.Data.Cols
	if axis==1||(axis==-1&&inC>1&&inR>=1&&!(inR==1&&inC>1)){outR=inR;outC=1;if keepDims{outC=1};resData=make([][]float64,outR);for i:=0;i<inR;i++{resData[i]=make([]float64,outC);if inC==0{if outC>0{resData[i][0]=0};continue};maxV:=input.Data.Data[i][0];for j:=1;j<inC;j++{if input.Data.Data[i][j]>maxV{maxV=input.Data.Data[i][j]}};resData[i][0]=maxV}} else if axis==0||(axis==-1&&inR>1&&inC>=1&&!(inC==1&&inR>1)){outR=1;outC=inC;if keepDims{outR=1};resData=make([][]float64,outR);resData[0]=make([]float64,outC);if inR==0{for j:=0;j<outC;j++{resData[0][j]=0}}else{for j:=0;j<inC;j++{maxV:=input.Data.Data[0][j];for i:=1;i<inR;i++{if input.Data.Data[i][j]>maxV{maxV=input.Data.Data[i][j]}};resData[0][j]=maxV}}} else if axis==-1{outR,outC=1,1;resData=make([][]float64,1);resData[0]=make([]float64,1);if inR==0||inC==0{resData[0][0]=0}else{maxV:=input.Data.Data[0][0];for i:=0;i<inR;i++{for j:=0;j<inC;j++{if input.Data.Data[i][j]>maxV{maxV=input.Data.Data[i][j]}}};resData[0][0]=maxV}} else{return nil,fmt.Errorf("TensorMax: unsupported axis %d",axis)}
	matRes,_:=NewMatrix(outR,outC,resData...);cfg:=&TensorConfig{RequiresGrad:input.RequiresGrad,Name:input.Name+"_max",Graph:graph};out,err:=NewTensor(matRes,cfg);if err!=nil{return nil,err}
	if out.RequiresGrad{out.Children=append(out.Children,input);out.BackwardFn=func(){
		if input.Grad==nil && input.RequiresGrad {input.Grad,_=NewMatrix(inR,inC)}
		if input.Grad != nil && out.Grad != nil {
			if axis==1||(axis==-1&&outR==inR&&outC==1){for i:=0;i<inR;i++{if inC==0{continue};maxV:=input.Data.Data[i][0];maxIdx:=0;for j:=1;j<inC;j++{if input.Data.Data[i][j]>maxV{maxV=input.Data.Data[i][j];maxIdx=j}};if i<out.Grad.Rows&&0<out.Grad.Cols{input.Grad.Data[i][maxIdx]+=out.Grad.Data[i][0]}}}else if axis==0||(axis==-1&&outC==inC&&outR==1){for j:=0;j<inC;j++{if inR==0{continue};maxV:=input.Data.Data[0][j];maxIdx:=0;for i:=1;i<inR;i++{if input.Data.Data[i][j]>maxV{maxV=input.Data.Data[i][j];maxIdx=i}};if 0<out.Grad.Rows&&j<out.Grad.Cols{input.Grad.Data[maxIdx][j]+=out.Grad.Data[0][j]}}}else if axis==-1{if inR>0&&inC>0&&out.Grad.Rows>0&&out.Grad.Cols>0{maxV:=input.Data.Data[0][0];max_i,max_j:=0,0;for i:=0;i<inR;i++{for j:=0;j<inC;j++{if input.Data.Data[i][j]>maxV{maxV=input.Data.Data[i][j];max_i,max_j=i,j}}};input.Grad.Data[max_i][max_j]+=out.Grad.Data[0][0]}}
		}
	};if graph!=nil && out.RequiresGrad {graph.addNode(out)}}; return out,nil
}
func TensorGather(input *Tensor, indices *Tensor, axis int) (*Tensor, error) {
	if input == nil || indices == nil { return nil, fmt.Errorf("input and indices tensors cannot be nil for Gather") }
    if axis != 0 { return nil, fmt.Errorf("TensorGather current implementation only supports axis=0 (row gathering)") }

    graph := input.Graph
    numIndicesRows := indices.Shape()[0]
    numIndicesCols := indices.Shape()[1]
    outputRows := numIndicesRows * numIndicesCols
    outputCols := input.Shape()[1]

    gatheredData := make([][]float64, outputRows)
    flatIndices := make([]int, outputRows)
    idxCounter := 0
    for i := 0; i < numIndicesRows; i++ {
        for j := 0; j < numIndicesCols; j++ {
            flatIndices[idxCounter] = int(indices.Data.Data[i][j])
            idxCounter++
        }
    }

    for i := 0; i < outputRows; i++ {
        originalRowIndex := flatIndices[i]
        gatheredData[i] = make([]float64, outputCols)
        if originalRowIndex >= 0 && originalRowIndex < input.Shape()[0] {
            copy(gatheredData[i], input.Data.Data[originalRowIndex])
        } else {
            fmt.Printf("Warning: TensorGather index %d out of bounds for input rows %d. Using zero vector.\n", originalRowIndex, input.Shape()[0])
        }
    }

    outputCfg := &TensorConfig{RequiresGrad: input.RequiresGrad, Name: fmt.Sprintf("Gather(%s)", input.Name), Graph: graph, DType: input.DType} // Preserve DType
    output, err := NewTensorFromData(gatheredData, outputCfg)
    if err != nil { return nil, fmt.Errorf("error creating output tensor for Gather: %w", err) }

    if input.RequiresGrad {
        output.Children = append(output.Children, input, indices)
        output.BackwardFn = func() {
            if input.Grad == nil && input.RequiresGrad { input.Grad, _ = NewMatrix(input.Shape()[0], input.Shape()[1]) }
            if input.Grad != nil && output.Grad != nil {
                for i := 0; i < outputRows; i++ {
                    originalRowIndex := flatIndices[i]
                    if originalRowIndex >= 0 && originalRowIndex < input.Shape()[0] {
                        if i < output.Grad.Rows {
                            for k := 0; k < outputCols; k++ {
                                input.Grad.Data[originalRowIndex][k] += output.Grad.Data[i][k]
                            }
                        }
                    }
                }
            }
        }
        if graph != nil && output.RequiresGrad { graph.addNode(output) }
    }
    return output, nil
}

// Helper for ifelse
func ifelse(condition bool, trueVal, falseVal string) string { if condition { return trueVal }; return falseVal }

// Legacy functions...
func LegacyCrossEntropyLoss(logits *Tensor, targets []int) *Tensor { result, _ := CrossEntropyLoss(logits, targets); return result }
func LegacyMSELoss(predictions, targets *Tensor) *Tensor { result, _ := MSELoss(predictions, targets); return result }
func LegacyReLU(a *Tensor) *Tensor { result, _ := ReLU(a); return result }
func LegacyGELU(a *Tensor) *Tensor { result, _ := GELU(a); return result }
func LegacySoftmax(a *Tensor) *Tensor { result, _ := TensorSoftmax(a, -1); return result }
func LegacyMatMul(a, b *Tensor) *Tensor { result, _ := MatMul(a, b); return result }
func LegacyAdd(a, b *Tensor) *Tensor { result, _ := Add(a, b); return result }
func LegacyNewTensor(data *Matrix, requiresGrad bool) *Tensor { t, _ := NewTensor(data, &TensorConfig{RequiresGrad: requiresGrad}); return t }
func LegacyNewRandomTensor(rows, cols int, requiresGrad bool) *Tensor { t, _ := NewRandomTensor(rows, cols, &TensorConfig{RequiresGrad: requiresGrad}); return t }
func LegacyNewZerosTensor(rows, cols int, requiresGrad bool) *Tensor { t, _ := NewZerosTensor(nil, rows, cols); t.SetRequiresGrad(requiresGrad); return t }

// DropoutTensor (global version, called by DropoutTensor struct method)
func DropoutTensor(input *Tensor, dropoutRate float64, isTraining bool, name string) (*Tensor, error) {
	if input == nil { return nil, fmt.Errorf("input tensor cannot be nil for DropoutTensor op") }
	if dropoutRate < 0.0 || dropoutRate >= 1.0 { return nil, fmt.Errorf("dropoutRate must be in [0,1)") }
	if !isTraining || dropoutRate == 0.0 { return input, nil }

	graph := input.Graph
	config := &TensorConfig{ RequiresGrad: input.RequiresGrad, Name: name, Graph: graph, DType: input.DType }

	resultData, _ := NewMatrix(input.Data.Rows, input.Data.Cols)
	dropoutMask, _ := NewMatrix(input.Data.Rows, input.Data.Cols)
	scale := 1.0 / (1.0 - dropoutRate)

	for i := 0; i < input.Data.Rows; i++ {
		for j := 0; j < input.Data.Cols; j++ {
			if rand.Float64() < dropoutRate {
				resultData.Data[i][j] = 0.0; dropoutMask.Data[i][j] = 0.0
			} else {
				resultData.Data[i][j] = input.Data.Data[i][j] * scale; dropoutMask.Data[i][j] = scale
			}
		}
	}
	result, err := NewTensor(resultData, config); if err != nil { return nil, err }

	if result.RequiresGrad {
		result.Children = append(result.Children, input)
		result.BackwardFn = func() {
			if input.Grad == nil && input.RequiresGrad { input.Grad, _ = NewMatrix(input.Data.Rows, input.Data.Cols) }
			if input.Grad != nil && result.Grad != nil {
				for i := 0; i < input.Grad.Rows; i++ {
					for j := 0; j < input.Grad.Cols; j++ {
						input.Grad.Data[i][j] += result.Grad.Data[i][j] * dropoutMask.Data[i][j]
					}
				}
			}
		}
		if graph != nil { graph.addNode(result) }
	}
	return result, nil
}

// ReLU activation
func ReLU(a *Tensor) (*Tensor, error) {
	if a == nil { return nil, fmt.Errorf("input tensor cannot be nil for ReLU") }
	graph := a.Graph
	config := &TensorConfig{RequiresGrad: a.RequiresGrad, Name: fmt.Sprintf("ReLU(%s)", a.Name), Graph: graph, DType: a.DType}
	
	resultData, _ := NewMatrix(a.Data.Rows, a.Data.Cols)
	for i := 0; i < a.Data.Rows; i++ {
		for j := 0; j < a.Data.Cols; j++ {
			if a.Data.Data[i][j] > 0 { resultData.Data[i][j] = a.Data.Data[i][j] } else { resultData.Data[i][j] = 0 }
		}
	}
	result, err := NewTensor(resultData, config); if err != nil {return nil, err}

	if result.RequiresGrad {
		result.Children = append(result.Children, a)
		result.BackwardFn = func() {
			if a.Grad == nil && a.RequiresGrad {a.Grad, _ = NewMatrix(a.Data.Rows, a.Data.Cols)}
			if a.Grad != nil && result.Grad != nil {
				for i := 0; i < a.Data.Rows; i++ { for j := 0; j < a.Data.Cols; j++ {
					if a.Data.Data[i][j] > 0 { a.Grad.Data[i][j] += result.Grad.Data[i][j] }
				} }
			}
		}
		if graph != nil {graph.addNode(result)}
	}
	return result, nil
}

// CrossEntropyLoss, MSELoss, Transpose, Sum, SliceColsTensor, ConcatenateColsTensor, ApplyAttentionMaskTensor, Clone
func CrossEntropyLoss(logits *Tensor, targets []int) (*Tensor, error) {
	if logits==nil||targets==nil{return nil,fmt.Errorf("inputs nil CE")};batchSize:=logits.Data.Rows;if len(targets)!=batchSize{return nil,fmt.Errorf("target/logit mismatch CE")};g:=logits.Graph
	cfg:=&TensorConfig{RequiresGrad:logits.RequiresGrad,Name:fmt.Sprintf("CEloss(%s)",logits.Name),Graph:g, DType:Float64};resData,_:=NewMatrix(1,1); lossV:=0.0
	softmaxOutputs := make([][]float64, batchSize); for i:=0;i<batchSize;i++{softmaxOutputs[i]=make([]float64,logits.Data.Cols)}

	for i:=0;i<batchSize;i++{if targets[i]<0||targets[i]>=logits.Data.Cols{return nil,fmt.Errorf("target idx %d out of bound %d",targets[i],logits.Data.Cols)};maxV:=-math.MaxFloat64;for j:=0;j<logits.Data.Cols;j++{if logits.Data.Data[i][j]>maxV{maxV=logits.Data.Data[i][j]}};sumExp:=0.0;for j:=0;j<logits.Data.Cols;j++{sVal:=math.Exp(logits.Data.Data[i][j]-maxV);softmaxOutputs[i][j]=sVal;sumExp+=sVal};logSumExp:=math.Log(sumExp)+maxV;lossV+=logSumExp-logits.Data.Data[i][targets[i]];if sumExp != 0 {for j:=0;j<logits.Data.Cols;j++{softmaxOutputs[i][j]/=sumExp}} }
	resData.Data[0][0]=lossV/float64(batchSize); res,err:=NewTensor(resData,cfg); if err != nil {return nil, err}
	if res.RequiresGrad{res.Children=append(res.Children,logits);res.BackwardFn=func(){
		if logits.Grad==nil && logits.RequiresGrad {logits.Grad,_=NewMatrix(logits.Data.Rows,logits.Data.Cols)}
		if logits.Grad != nil && res.Grad != nil {
			for i:=0;i<batchSize;i++{for j:=0;j<logits.Data.Cols;j++{grad:=softmaxOutputs[i][j];if j==targets[i]{grad-=1.0};logits.Grad.Data[i][j]+=grad*res.Grad.Data[0][0]/float64(batchSize)}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func MSELoss(preds *Tensor, tgts *Tensor) (*Tensor, error) {
	if preds==nil||tgts==nil{return nil,fmt.Errorf("inputs nil MSE")};if preds.Data.Rows!=tgts.Data.Rows||preds.Data.Cols!=tgts.Data.Cols{return nil,fmt.Errorf("dim mismatch MSE")};g:=preds.Graph
	cfg:=&TensorConfig{RequiresGrad:preds.RequiresGrad,Name:fmt.Sprintf("MSEloss(%s)",preds.Name),Graph:g,DType:Float64};resData,_:=NewMatrix(1,1);totalE:=float64(preds.Data.Rows*preds.Data.Cols);if totalE == 0 {return NewZerosTensor(cfg,1,1)};lossV:=0.0
	for i:=0;i<preds.Data.Rows;i++{for j:=0;j<preds.Data.Cols;j++{diff:=preds.Data.Data[i][j]-tgts.Data.Data[i][j];lossV+=diff*diff}};lossV/=totalE;resData.Data[0][0]=lossV
	res,err:=NewTensor(resData,cfg); if err != nil {return nil, err}
	if res.RequiresGrad{res.Children=append(res.Children,preds);res.BackwardFn=func(){
		if preds.Grad==nil && preds.RequiresGrad {preds.Grad,_=NewMatrix(preds.Data.Rows,preds.Data.Cols)}
		if preds.Grad != nil && res.Grad != nil && totalE != 0 {
			for i:=0;i<preds.Data.Rows;i++{for j:=0;j<preds.Data.Cols;j++{diff:=2.0*(preds.Data.Data[i][j]-tgts.Data.Data[i][j])/totalE;preds.Grad.Data[i][j]+=diff*res.Grad.Data[0][0]}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func TensorTranspose(a *Tensor) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil Transpose")};g:=a.Graph;cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:a.Name+"_T",Graph:g, DType:a.DType}
	datT,_:=Transpose(a.Data);res,_:=NewTensor(datT,cfg)
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
		if a.Grad != nil && res.Grad != nil {
			for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{a.Grad.Data[i][j]+=res.Grad.Data[j][i]}}
		}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func Sum(a *Tensor) (*Tensor, error) {
	if a==nil{return nil,fmt.Errorf("input nil Sum")};g:=a.Graph;cfg:=&TensorConfig{RequiresGrad:a.RequiresGrad,Name:a.Name+"_sum",Graph:g, DType:Float64}
	resD,_:=NewMatrix(1,1);sumV:=0.0;for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{sumV+=a.Data.Data[i][j]}};resD.Data[0][0]=sumV
	res,_:=NewTensor(resD,cfg)
	if res.RequiresGrad{res.Children=append(res.Children,a);res.BackwardFn=func(){
		if a.Grad==nil && a.RequiresGrad {a.Grad,_=NewMatrix(a.Data.Rows,a.Data.Cols)}
		if a.Grad != nil && res.Grad != nil {
		for i:=0;i<a.Data.Rows;i++{for j:=0;j<a.Data.Cols;j++{a.Grad.Data[i][j]+=res.Grad.Data[0][0]}}}
	};if g!=nil{g.addNode(res)}}; return res,nil
}
func SliceColsTensor(input *Tensor, startCol, numCols int, name string) (*Tensor, error) {
	if input == nil { return nil, fmt.Errorf("input tensor cannot be nil") }; if numCols < 0 { return nil, fmt.Errorf("numCols must be non-negative") }
	endCol := startCol + numCols; if startCol < 0 || endCol > input.Data.Cols || startCol > endCol { return nil, fmt.Errorf("col slice out of bounds") }
	slicedData, err := sliceCols(input.Data, startCol, endCol); if err != nil { return nil, err }
	cfg := &TensorConfig{ RequiresGrad: input.RequiresGrad, Name: name, Graph: input.Graph, DType: input.DType }; output, err := NewTensor(slicedData, cfg); if err != nil { return nil, err }
	if input.RequiresGrad { output.Children = append(output.Children, input); output.BackwardFn = func() {
		if input.Grad == nil && input.RequiresGrad { input.Grad, _ = NewMatrix(input.Data.Rows, input.Data.Cols) }
		if input.Grad != nil && output.Grad != nil {
			for i := 0; i < output.Grad.Rows; i++ { for j := 0; j < output.Grad.Cols; j++ { input.Grad.Data[i][startCol+j] += output.Grad.Data[i][j] } }
		}
	}; if input.Graph != nil && output.RequiresGrad { input.Graph.addNode(output) } }; return output, nil
}
func ConcatenateColsTensor(tensors []*Tensor, name string) (*Tensor, error) {
	if len(tensors)==0{return nil,fmt.Errorf("empty list for Concat")};dataM:=make([]*Matrix,len(tensors));anyReqGrad:=false;var g *ComputationGraph; if tensors[0] != nil {g=tensors[0].Graph}
	var commonDType DataType = Float64; if len(tensors)>0 && tensors[0]!=nil {commonDType = tensors[0].DType}
	for i,t:=range tensors{if t==nil{return nil,fmt.Errorf("nil tensor at %d",i)};dataM[i]=t.Data;if t.RequiresGrad{anyReqGrad=true};if g==nil && t.Graph != nil {g=t.Graph}; if t.DType != commonDType {return nil, fmt.Errorf("dtype mismatch in Concat")}}
	concatD,err:=concatenateCols(dataM);if err!=nil{return nil,err}
	cfg:=&TensorConfig{RequiresGrad:anyReqGrad,Name:name,Graph:g,DType:commonDType};out,err:=NewTensor(concatD,cfg);if err!=nil{return nil,err}
	if anyReqGrad{ childrenReqGrad:=make([]*Tensor,0);for _,t:=range tensors{if t.RequiresGrad{childrenReqGrad=append(childrenReqGrad,t)}}; out.Children=childrenReqGrad
		out.BackwardFn=func(){currSC:=0;for _,inT:=range tensors{if inT.RequiresGrad{if inT.Grad==nil && inT.RequiresGrad {inT.Grad,_=NewMatrix(inT.Data.Rows,inT.Data.Cols)}; if inT.Grad != nil && out.Grad != nil {nC:=inT.Data.Cols;for i:=0;i<out.Grad.Rows;i++{for j:=0;j<nC;j++{inT.Grad.Data[i][j]+=out.Grad.Data[i][currSC+j]}}}};currSC+=inT.Data.Cols}}
	;if g!=nil && out.RequiresGrad {g.addNode(out)}}; return out,nil
}
func ApplyAttentionMaskTensor(scores *Tensor, maskTensor *Tensor, maskValue float64, name string) (*Tensor, error) {
	if scores==nil||maskTensor==nil{return nil,fmt.Errorf("inputs nil for ApplyMask")};if scores.Data.Rows!=maskTensor.Data.Rows||scores.Data.Cols!=maskTensor.Data.Cols{return nil,fmt.Errorf("shape mismatch ApplyMask")}
	g:=scores.Graph;cfg:=&TensorConfig{RequiresGrad:scores.RequiresGrad,Name:name,Graph:g,DType:scores.DType};resD,_:=NewMatrix(scores.Data.Rows,scores.Data.Cols)
	for i:=0;i<scores.Data.Rows;i++{for j:=0;j<scores.Data.Cols;j++{if maskTensor.Data.Data[i][j]==0{resD.Data[i][j]=maskValue}else{resD.Data[i][j]=scores.Data.Data[i][j]}}}
	res,err:=NewTensor(resD,cfg);if err!=nil{return nil,err}
	if scores.RequiresGrad{res.Children=append(res.Children,scores);res.BackwardFn=func(){
		if scores.Grad==nil && scores.RequiresGrad {scores.Grad,_=NewMatrix(scores.Data.Rows,scores.Data.Cols)}
		if scores.Grad != nil && res.Grad != nil {
			for i:=0;i<scores.Grad.Rows;i++{for j:=0;j<scores.Grad.Cols;j++{if maskTensor.Data.Data[i][j]==1{scores.Grad.Data[i][j]+=res.Grad.Data[i][j]}}}}
		}
	};if g!=nil && res.RequiresGrad {g.addNode(res)}}; return res,nil
}
func (t *Tensor) Clone() (*Tensor, error) {
	if t==nil{return nil,fmt.Errorf("cannot clone nil")};dC,err:=t.Data.Clone();if err!=nil{return nil,err};var gC *Matrix;if t.Grad!=nil{gC,err=t.Grad.Clone();if err!=nil{return nil,err}}
	return &Tensor{Data:dC,Grad:gC,RequiresGrad:t.RequiresGrad,Name:t.Name+"_clone",Graph:t.Graph,dtype:t.DType,shape:append([]int(nil),t.shape...)},nil
}

[end of pkg/autodiff/autodiff.go]
