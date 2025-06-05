package autodiff

import (
	"math"
	"fmt"
)

// MoELayer is the main layer that orchestrates token routing and processing by experts.
// This type was moved from pkg/moe/moe_layer.go to pkg/autodiff/moe_layer.go
// to break an import cycle. It now uses local Expert and Router types.
type MoELayer struct {
	Config        MoELayerConfig // Uses MoELayerConfig from moe_types.go
	Experts       []*Expert      // Uses Expert from moe_components.go
	Router        *Router        // Uses Router from moe_components.go
	AuxiliaryLoss *Tensor
}

// NewMoELayer constructor
func NewMoELayer(config MoELayerConfig, requiresGrad bool, graph *ComputationGraph) *MoELayer {
	router := NewRouter(config.ModelDim, config.NumExperts, requiresGrad, graph) // Local NewRouter
	experts := make([]*Expert, config.NumExperts)                                // Local Expert
	for i := 0; i < config.NumExperts; i++ {
		expertHiddenDim := config.HiddenDim
		if expertHiddenDim == 0 {
			expertHiddenDim = config.ModelDim * 4
		}
		activationFunc := config.Activation
		if activationFunc == nil {
			activationFunc = GELU // Default to GELU if nil
		}
		experts[i] = NewExpert(config.ModelDim, expertHiddenDim, config.ModelDim, activationFunc, requiresGrad, graph) // Local NewExpert
	}
	auxLossData, _ := NewMatrix(1, 1) // Assuming NewMatrix is available (it is in autodiff)
	auxLoss, _ := NewTensor(auxLossData, &TensorConfig{RequiresGrad: false, Name: "moe_aux_loss_field", Graph: graph})

	return &MoELayer{Config: config, Experts: experts, Router: router, AuxiliaryLoss: auxLoss}
}

// GetParameters method
func (ml *MoELayer) GetParameters() []*Tensor {
	params := ml.Router.GetParameters()
	for _, expert := range ml.Experts {
		params = append(params, expert.GetParameters()...)
	}
	return params
}

// GetAuxiliaryLoss returns the auxiliary loss tensor for the MoE layer.
// This method allows MoELayer to satisfy the AuxiliaryLossProvider interface.
func (ml *MoELayer) GetAuxiliaryLoss() *Tensor {
	return ml.AuxiliaryLoss
}

// Forward method for MoELayer
func (ml *MoELayer) Forward(inputTokens *Tensor, isTraining bool) (*Tensor, error) {
	if ml.Config.TopK <= 0 {
		fmt.Println("Warning: MoELayer TopK is <= 0, bypassing MoE layer.")
		// Ensure NewMatrixZeros is available or use NewTensor with zeroed data
		zeroData, _ := NewMatrix(1,1) // Create a 1x1 zero matrix data
		ml.AuxiliaryLoss, _ = NewTensor(zeroData, &TensorConfig{Graph: inputTokens.Graph, Name: "bypassed_aux_loss"})
		return inputTokens, nil
	}

	graph := inputTokens.Graph
	numTotalTokens := inputTokens.Shape()[0]

	routerLogits, err := ml.Router.Forward(inputTokens)
	if err != nil {
		return nil, fmt.Errorf("router forward failed: %w", err)
	}

	if ml.Config.NoisyRouting && isTraining {
		// Placeholder for noisy routing implementation
		// noise, _ := NewNormalTensor(routerLogits.Shape(), 0, 1e-2, &TensorConfig{Graph: graph, Name: "routing_noise"})
		// routerLogits, _ = Add(routerLogits, noise)
		fmt.Println("Note: Noisy routing is enabled but NewNormalTensor/Matrix is a TODO, skipping noise application.")
	}

	routerProbs, err := TensorSoftmax(routerLogits, -1)
	if err != nil {
		return nil, fmt.Errorf("router softmax failed: %w", err)
	}

	topKRouterProbs, topKExpertIndices, err := TensorTopK(routerProbs, ml.Config.TopK, 1, true)
	if err != nil {
		fmt.Printf("Error during TopK selection (K=%d): %v. MoE will be passthrough.\n", ml.Config.TopK, err)
		zeroData, _ := NewMatrix(1,1)
		ml.AuxiliaryLoss, _ = NewTensor(zeroData, &TensorConfig{Graph: graph, Name: "topk_err_aux_loss"})
		return inputTokens, nil
	}

	finalCombinedOutputConfig := &TensorConfig{Graph: graph, RequiresGrad: inputTokens.RequiresGrad, Name: "moe_final_output"}
	finalCombinedOutput, _ := NewZerosTensor(finalCombinedOutputConfig, inputTokens.Shape()...)

	expertAssignmentMasksForLBLoss := make([]*Tensor, ml.Config.NumExperts)


	for kIdx := 0; kIdx < ml.Config.TopK; kIdx++ {
		currentExpertIndicesSlice, errSlice := TensorSlice(topKExpertIndices, []*SliceArg{{Start:0, End:numTotalTokens}, {Start:kIdx, End:kIdx+1}}, fmt.Sprintf("topk_idx_k%d", kIdx))
		if errSlice != nil {return nil, fmt.Errorf("slicing topKExpertIndices for k_idx %d: %w", kIdx, errSlice)}

		currentRouterProbsSlice, errSlice := TensorSlice(topKRouterProbs, []*SliceArg{{Start:0, End:numTotalTokens}, {Start:kIdx, End:kIdx+1}}, fmt.Sprintf("topk_prob_k%d", kIdx))
		if errSlice != nil {return nil, fmt.Errorf("slicing topKRouterProbs for k_idx %d: %w", kIdx, errSlice)}


		for e := 0; e < ml.Config.NumExperts; e++ {
			activeExpertMaskComparative, errEq := TensorEqualScalar(currentExpertIndicesSlice, float64(e))
			if errEq != nil { fmt.Printf("Error TensorEqualScalar expert %d, k_idx %d: %v\n", e, kIdx, errEq); continue }

			activeExpertMaskF, errCast := TensorCast(activeExpertMaskComparative, Float64)
			if errCast != nil { fmt.Printf("Error TensorCast expert %d, k_idx %d: %v\n", e, kIdx, errCast); continue }

			if kIdx == 0 {
				if expertAssignmentMasksForLBLoss[e] == nil {
					expertAssignmentMasksForLBLoss[e] = activeExpertMaskF
				}
			}

			currentExpertInput, errOp := Multiply(inputTokens, activeExpertMaskF)
			if errOp != nil { fmt.Printf("Error Multiply expert input expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }

			expertOutput, errOp := ml.Experts[e].Forward(currentExpertInput)
			if errOp != nil { fmt.Printf("Error Expert %d Forward, k_idx %d: %v\n", e, kIdx, errOp); continue }

			gatingWeight, errOp := Multiply(currentRouterProbsSlice, activeExpertMaskF)
			if errOp != nil { fmt.Printf("Error Multiply gating weight expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }

			weightedExpertOutput, errOp := Multiply(expertOutput, gatingWeight)
			if errOp != nil { fmt.Printf("Error Multiply weighted output expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }

			tempFinalCombinedOutput, errOp := Add(finalCombinedOutput, weightedExpertOutput)
			if errOp != nil { fmt.Printf("Error Add final output expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }
			finalCombinedOutput = tempFinalCombinedOutput
		}
	}

	auxLossConfig := &TensorConfig{Graph: graph, Name: "total_aux_loss_val", RequiresGrad: true}
	totalAuxLossZData, _ := NewMatrix(1,1)
	totalAuxLoss, _ := NewTensor(totalAuxLossZData, auxLossConfig)


	if isTraining {
		if ml.Config.RouterZLossCoeff > 0 {
			logSumExpRouterLogits, errLSE := TensorLogSumExp(routerLogits, 1, false)
			if errLSE != nil { fmt.Printf("Error LogSumExp for Router Z-Loss: %v.\n", errLSE)
			} else {
				squaredLogSumExp, errSq := TensorSquare(logSumExpRouterLogits)
				if errSq != nil { fmt.Printf("Error Square for Router Z-Loss: %v.\n", errSq)
				} else {
					routerZLossValue, errMean := TensorMean(squaredLogSumExp, -1, false)
					if errMean != nil { fmt.Printf("Error Mean for Router Z-Loss: %v.\n", errMean)
					} else {
						routerZLoss, errMulS := ScalarMultiply(routerZLossValue, ml.Config.RouterZLossCoeff)
						if errMulS != nil { fmt.Printf("Error ScalarMultiply for Router Z-Loss: %v.\n", errMulS)
						} else {
							var errAddAux error; totalAuxLoss, errAddAux = Add(totalAuxLoss, routerZLoss)
							if errAddAux != nil {fmt.Printf("Error adding ZLoss to TotalAuxLoss: %v\n", errAddAux)}
						}
					}
				}
			}
		}

		if ml.Config.LoadBalanceLossCoeff > 0 {
			var P_i_tensor, f_i_tensor *Tensor
			var errLBLoss error

			P_i_tensor, errLBLoss = TensorMean(routerProbs, 0, false)
			if errLBLoss != nil { fmt.Printf("Error calculating P_i for LB Loss: %v.\n", errLBLoss)
			} else {
				f_i_data_list := make([]float64, ml.Config.NumExperts)
				validFi := true
				if len(expertAssignmentMasksForLBLoss) == ml.Config.NumExperts {
					for e := 0; e < ml.Config.NumExperts; e++ {
						if expertAssignmentMasksForLBLoss[e] == nil {
							f_i_data_list[e] = 0.0
							continue
						}
						sumMaskTensor, errSumMask := TensorSum(expertAssignmentMasksForLBLoss[e], -1, false)
						if errSumMask != nil { validFi = false; break }
						if numTotalTokens > 0 {
							f_i_data_list[e] = sumMaskTensor.Data()[0][0] / float64(numTotalTokens)
						} else {
							f_i_data_list[e] = 0.0
						}
					}
				} else { validFi = false }


				if validFi {
					f_i_matrix_data := NewMatrixFromSlice(f_i_data_list, 1, ml.Config.NumExperts)
					f_i_tensor_cfg := &TensorConfig{Graph: graph, Name: "f_i_fractions"}
					f_i_tensor, _ = NewTensor(f_i_matrix_data, f_i_tensor_cfg)
				} else {
					f_i_tensor = P_i_tensor
					fmt.Println("Using P_i for f_i in Load Balancing Loss (issues with masks or K>1).")
				}


				if f_i_tensor != nil && P_i_tensor != nil {
					products, errProd := Multiply(f_i_tensor, P_i_tensor)
					if errProd != nil { fmt.Printf("Error calculating products for LB Loss: %v.\n", errProd)
					} else {
						sumOverExperts, errSumProd := TensorSum(products, -1, false)
						if errSumProd != nil { fmt.Printf("Error summing products for LB Loss: %v.\n", errSumProd)
						} else {
							finalLBCoeff := ml.Config.LoadBalanceLossCoeff * float64(ml.Config.NumExperts)
							loadBalanceLossVal, errLBMul := ScalarMultiply(sumOverExperts, finalLBCoeff)
							if errLBMul != nil { fmt.Printf("Error ScalarMultiply for LB Loss: %v.\n", errLBMul)
							} else {
								var addLBErr error; totalAuxLoss, addLBErr = Add(totalAuxLoss, loadBalanceLossVal)
								if addLBErr != nil { fmt.Printf("Error adding LB Loss to TotalAuxLoss: %v\n", addLBErr) }
							}
						}
					}
				}
			}
		}
		ml.AuxiliaryLoss = totalAuxLoss
	} else {
		zeroData, _ := NewMatrix(1,1)
		ml.AuxiliaryLoss, _ = NewTensor(zeroData, &TensorConfig{Graph: graph, Name: "no_train_aux_loss", RequiresGrad: false})
	}

	return finalCombinedOutput, nil
}
