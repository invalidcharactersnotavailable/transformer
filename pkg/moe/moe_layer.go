package moe

import (
	"math"
	"fmt"

	"github.com/transformer_reorganized/pkg/autodiff"
)

// MoELayerConfig holds the configuration for an MoELayer.
type MoELayerConfig struct {
	ModelDim             int
	NumExperts           int
	HiddenDim            int
	TopK                 int
	CapacityFactor       float64
	NoisyRouting         bool
	RouterZLossCoeff     float64 // Added
	LoadBalanceLossCoeff float64 // Added
	Activation           func(*autodiff.Tensor) (*autodiff.Tensor, error)
}

// MoELayer is the main layer that orchestrates token routing and processing by experts.
type MoELayer struct {
	Config        MoELayerConfig // Now includes loss coeffs
	Experts       []*Expert
	Router        *Router
	AuxiliaryLoss *autodiff.Tensor
}

// NewMoELayer constructor
func NewMoELayer(config MoELayerConfig, requiresGrad bool, graph *autodiff.ComputationGraph) *MoELayer {
	router := NewRouter(config.ModelDim, config.NumExperts, requiresGrad, graph)
	experts := make([]*Expert, config.NumExperts)
	for i := 0; i < config.NumExperts; i++ {
		expertHiddenDim := config.HiddenDim; if expertHiddenDim == 0 { expertHiddenDim = config.ModelDim * 4 }
		activationFunc := config.Activation; if activationFunc == nil { activationFunc = autodiff.GELU }
		experts[i] = NewExpert(config.ModelDim, expertHiddenDim, config.ModelDim, activationFunc, requiresGrad, graph)
	}
	// AuxiliaryLoss is initialized per forward pass if training, or to a non-grad zero tensor.
	// So, no need to initialize it with requiresGrad here.
	auxLossData, _ := autodiff.NewMatrix(1,1);
	auxLoss, _ := autodiff.NewTensor(auxLossData, &autodiff.TensorConfig{RequiresGrad: false, Name: "moe_aux_loss_field", Graph: graph})

	return &MoELayer{ Config: config, Experts: experts, Router: router, AuxiliaryLoss: auxLoss }
}

// GetParameters method
func (ml *MoELayer) GetParameters() []*autodiff.Tensor {
	params := ml.Router.GetParameters()
	for _, expert := range ml.Experts { params = append(params, expert.GetParameters()...) }
	return params
}

// Forward method for MoELayer
func (ml *MoELayer) Forward(inputTokens *autodiff.Tensor, isTraining bool) (*autodiff.Tensor, error) {
	if ml.Config.TopK <= 0 {
		fmt.Println("Warning: MoELayer TopK is <= 0, bypassing MoE layer.")
		ml.AuxiliaryLoss, _ = autodiff.NewTensor(autodiff.NewMatrixZeros(1,1), &autodiff.TensorConfig{Graph: inputTokens.Graph, Name: "bypassed_aux_loss"})
		return inputTokens, nil
	}

	graph := inputTokens.Graph
	numTotalTokens := inputTokens.Shape()[0]

	routerLogits, err := ml.Router.Forward(inputTokens); if err != nil { return nil, fmt.Errorf("router forward failed: %w", err) }

	if ml.Config.NoisyRouting && isTraining {
		fmt.Println("Note: Noisy routing is enabled but NewNormalTensor/Matrix is a TODO, skipping noise application.")
	}

	routerProbs, err := autodiff.TensorSoftmax(routerLogits, -1); if err != nil { return nil, fmt.Errorf("router softmax failed: %w", err) }

	topKRouterProbs, topKExpertIndices, err := autodiff.TensorTopK(routerProbs, ml.Config.TopK, 1, true)
	if err != nil {
		ml.AuxiliaryLoss, _ = autodiff.NewTensor(autodiff.NewMatrixZeros(1,1), &autodiff.TensorConfig{Graph: graph, Name: "topk_err_aux_loss"})
		return nil, fmt.Errorf("TensorTopK failed (K=%d): %w", ml.Config.TopK, err)
	}

	finalCombinedOutputConfig := &autodiff.TensorConfig{Graph: graph, RequiresGrad: inputTokens.RequiresGrad, Name: "moe_final_output"}
	finalCombinedOutput, _ := autodiff.NewZerosTensor(finalCombinedOutputConfig, inputTokens.Shape()...)

	expertAssignmentMasksForLBLoss := make([]*autodiff.Tensor, ml.Config.NumExperts)

	for kIdx := 0; kIdx < ml.Config.TopK; kIdx++ {
		currentExpertIndicesSlice, errSlice := autodiff.TensorSlice(topKExpertIndices, []*autodiff.SliceArg{{Start:0, End:numTotalTokens}, {Start:kIdx, End:kIdx+1}}, fmt.Sprintf("topk_idx_k%d", kIdx))
		if errSlice != nil {return nil, fmt.Errorf("slicing topKExpertIndices for k_idx %d: %w", kIdx, errSlice)}

		currentRouterProbsSlice, errSlice := autodiff.TensorSlice(topKRouterProbs, []*autodiff.SliceArg{{Start:0, End:numTotalTokens}, {Start:kIdx, End:kIdx+1}}, fmt.Sprintf("topk_prob_k%d", kIdx))
		if errSlice != nil {return nil, fmt.Errorf("slicing topKRouterProbs for k_idx %d: %w", kIdx, errSlice)}

		for e := 0; e < ml.Config.NumExperts; e++ {
			activeExpertMaskComparative, errEq := autodiff.TensorEqualScalar(currentExpertIndicesSlice, float64(e))
			if errEq != nil { return nil, fmt.Errorf("TensorEqualScalar failed for expert %d, k_idx %d: %w", e, kIdx, errEq) }

			activeExpertMaskF, errCast := autodiff.TensorCast(activeExpertMaskComparative, autodiff.Float64)
			if errCast != nil { return nil, fmt.Errorf("TensorCast failed for expert %d, k_idx %d: %w", e, kIdx, errCast) }

			if kIdx == 0 { // For LB Loss, only consider top-1 choice's mask for simplicity
				if expertAssignmentMasksForLBLoss[e] == nil {
					expertAssignmentMasksForLBLoss[e] = activeExpertMaskF
				} else {
					// This case (kIdx=0 but mask already exists) implies an error or change in logic.
					// For safety, let's assume this shouldn't happen if K=1 logic is distinct or TopK=1 used for LB.
					// If K>1 and we want to sum masks for LB loss from all K choices, this Add is needed.
					// However, current LB loss is simplified and might not use this sum correctly.
					// For now, this logic is fine for K=1, for K>1 LB loss is approximated.
					// expertAssignmentMasksForLBLoss[e], _ = autodiff.Add(expertAssignmentMasksForLBLoss[e], activeExpertMaskF)
				}
			}

			currentExpertInput, errOp := autodiff.Multiply(inputTokens, activeExpertMaskF)
			if errOp != nil { return nil, fmt.Errorf("Multiply currentExpertInput failed for expert %d, k_idx %d: %w", e, kIdx, errOp) }

			expertOutput, errOp := ml.Experts[e].Forward(currentExpertInput)
			if errOp != nil { return nil, fmt.Errorf("Expert %d Forward failed for k_idx %d: %w", e, kIdx, errOp) }

			gatingWeight, errOp := autodiff.Multiply(currentRouterProbsSlice, activeExpertMaskF)
			if errOp != nil { return nil, fmt.Errorf("Multiply gatingWeight failed for expert %d, k_idx %d: %w", e, kIdx, errOp) }

			weightedExpertOutput, errOp := autodiff.Multiply(expertOutput, gatingWeight)
			if errOp != nil { return nil, fmt.Errorf("Multiply weightedExpertOutput failed for expert %d, k_idx %d: %w", e, kIdx, errOp) }

			tempFinalCombinedOutput, errOp := autodiff.Add(finalCombinedOutput, weightedExpertOutput)
			if errOp != nil { return nil, fmt.Errorf("Add tempFinalCombinedOutput failed for expert %d, k_idx %d: %w", e, kIdx, errOp) }
			finalCombinedOutput = tempFinalCombinedOutput
		}
	}

	// Initialize totalAuxLoss as a graph-connected zero tensor that requires grad for accumulation
	// Ensure it's on the same graph as other tensors involved in loss.
	auxLossConfig := &autodiff.TensorConfig{Graph: graph, Name: "total_aux_loss_val", RequiresGrad: true}
	totalAuxLoss, _ := autodiff.NewTensor(autodiff.NewMatrixZeros(1,1), auxLossConfig)


	if isTraining {
		// Router Z-Loss (only if coefficient is positive)
		if ml.Config.RouterZLossCoeff > 0 {
			logSumExpRouterLogits, errLSE := autodiff.TensorLogSumExp(routerLogits, 1, false)
			if errLSE != nil { return nil, fmt.Errorf("TensorLogSumExp for Router Z-Loss failed: %w", errLSE)
			} else {
				squaredLogSumExp, errSq := autodiff.TensorSquare(logSumExpRouterLogits)
				if errSq != nil { return nil, fmt.Errorf("TensorSquare for Router Z-Loss failed: %w", errSq)
				} else {
					routerZLossValue, errMean := autodiff.TensorMean(squaredLogSumExp, -1, false)
					if errMean != nil { return nil, fmt.Errorf("TensorMean for Router Z-Loss failed: %w", errMean)
					} else {
						routerZLoss, errMulS := autodiff.ScalarMultiply(routerZLossValue, ml.Config.RouterZLossCoeff)
						if errMulS != nil { return nil, fmt.Errorf("ScalarMultiply for Router Z-Loss failed: %w", errMulS)
						} else {
							var errAddAux error; totalAuxLoss, errAddAux = autodiff.Add(totalAuxLoss, routerZLoss)
							if errAddAux != nil { return nil, fmt.Errorf("failed to add Router Z-Loss to TotalAuxLoss: %w", errAddAux) }
						}
					}
				}
			}
		}

		// Load Balancing Loss (only if coefficient is positive)
		if ml.Config.LoadBalanceLossCoeff > 0 {
			var P_i_tensor, f_i_tensor *autodiff.Tensor; var errLBLoss error
			P_i_tensor, errLBLoss = autodiff.TensorMean(routerProbs, 0, false)
			if errLBLoss != nil { return nil, fmt.Errorf("calculating P_i for Load Balancing Loss failed: %w", errLBLoss)
			} else {
				f_i_tensor = P_i_tensor // Default if cannot calculate f_i from masks

				// This f_i calculation from masks is primarily for TopK=1.
				if ml.Config.TopK == 1 && len(expertAssignmentMasksForLBLoss) == ml.Config.NumExperts {
					f_i_data_list := make([]float64, ml.Config.NumExperts)
					validFi := true
					for e := 0; e < ml.Config.NumExperts; e++ {
						if expertAssignmentMasksForLBLoss[e] == nil { validFi = false; break }
						sumMaskTensor, errSumMask := autodiff.TensorMean(expertAssignmentMasksForLBLoss[e], -1, false)
						if errSumMask != nil { return nil, fmt.Errorf("TensorMean for sumMaskTensor (LB Loss) failed for expert %d: %w", e, errSumMask) }

						countTensorScaled, errScale := autodiff.ScalarMultiply(sumMaskTensor, float64(expertAssignmentMasksForLBLoss[e].Shape()[0]))
						if errScale != nil { return nil, fmt.Errorf("ScalarMultiply for countTensorScaled (LB Loss) failed for expert %d: %w", e, errScale) }
						f_i_data_list[e] = countTensorScaled.Data()[0][0]
					}

					if validFi && numTotalTokens > 0 {
						f_i_matrix_data := autodiff.NewMatrixFromSlice(f_i_data_list, 1, ml.Config.NumExperts)
						f_i_tensor_unnormalized_cfg := &autodiff.TensorConfig{Graph: graph, Name: "f_i_counts"}
						f_i_tensor_unnormalized, _ := autodiff.NewTensor(f_i_matrix_data, f_i_tensor_unnormalized_cfg)
						f_i_tensor, _ = autodiff.ScalarMultiply(f_i_tensor_unnormalized, 1.0/float64(numTotalTokens))
					} else { fmt.Println("Using P_i for f_i in Load Balancing Loss (issues with masks or zero tokens).") }
				} else if ml.Config.TopK > 1 {
					fmt.Println("Using P_i for f_i in Load Balancing Loss for TopK > 1 (more accurate f_i TODO).")
				}

				if f_i_tensor != nil && P_i_tensor != nil { // Ensure both are valid before Multiply
					products, errProd := autodiff.Multiply(f_i_tensor, P_i_tensor)
					if errProd != nil { return nil, fmt.Errorf("calculating products for Load Balancing Loss failed: %w", errProd)
					} else {
						sumOverExperts, errSumProd := autodiff.TensorMean(products, -1, false)
						if errSumProd != nil { return nil, fmt.Errorf("summing products for Load Balancing Loss failed: %w", errSumProd)
						} else {
							finalLBCoeff := ml.Config.LoadBalanceLossCoeff * float64(ml.Config.NumExperts*ml.Config.NumExperts)
							loadBalanceLossVal, errLBMul := autodiff.ScalarMultiply(sumOverExperts, finalLBCoeff)
							if errLBMul != nil { return nil, fmt.Errorf("ScalarMultiply for Load Balancing Loss value failed: %w", errLBMul)
							} else {
								var addLBErr error; totalAuxLoss, addLBErr = autodiff.Add(totalAuxLoss, loadBalanceLossVal)
								if addLBErr != nil { return nil, fmt.Errorf("failed to add Load Balancing Loss to TotalAuxLoss: %w", addLBErr) }
							}
						}
					}
				}
			}
		}
		ml.AuxiliaryLoss = totalAuxLoss
	} else {
		ml.AuxiliaryLoss, _ = autodiff.NewTensor(autodiff.NewMatrixZeros(1,1), &autodiff.TensorConfig{Graph: graph, Name: "no_train_aux_loss"})
	}
	return finalCombinedOutput, nil
}
