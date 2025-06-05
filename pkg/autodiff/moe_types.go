package autodiff

// MoELayerConfig holds the configuration for an MoELayer.
// This type was moved from pkg/moe/moe_layer.go to pkg/autodiff/moe_types.go
// to break an import cycle.
type MoELayerConfig struct {
	ModelDim             int
	NumExperts           int
	HiddenDim            int
	TopK                 int
	CapacityFactor       float64
	NoisyRouting         bool
	RouterZLossCoeff     float64
	LoadBalanceLossCoeff float64
	Activation           func(*Tensor) (*Tensor, error) // Assuming autodiff.Tensor
}
