package autodiff

import (
	"math"
	"fmt"
)

// MoELayerConfig holds the configuration for an MoELayer.
type MoELayerConfig struct {
	ModelDim             int
	NumExperts           int
	HiddenDim            int
	TopK                 int
	CapacityFactor       float64
	NoisyRouting         bool
	RouterZLossCoeff     float64
	LoadBalanceLossCoeff float64
	Activation           func(*Tensor) (*Tensor, error)
}

// MoELayer is the main layer that orchestrates token routing and processing by experts.
type MoELayer struct {
	Config        MoELayerConfig
	Experts       []*Expert // Assumes Expert is now in package autodiff
	Router        *Router   // Assumes Router is now in package autodiff
	AuxiliaryLoss *Tensor
}

// NewMoELayer constructor
func NewMoELayer(config MoELayerConfig, requiresGrad bool, graph *ComputationGraph) *MoELayer {
	router := NewRouter(config.ModelDim, config.NumExperts, requiresGrad, graph) // NewRouter is now local
	experts := make([]*Expert, config.NumExperts)
	for i := 0; i < config.NumExperts; i++ {
		expertHiddenDim := config.HiddenDim; if expertHiddenDim == 0 { expertHiddenDim = config.ModelDim * 4 }
		activationFunc := config.Activation; if activationFunc == nil { activationFunc = GELU } // GELU is local
		experts[i] = NewExpert(config.ModelDim, expertHiddenDim, config.ModelDim, activationFunc, requiresGrad, graph) // NewExpert is local
	}
	auxLossData, _ := NewMatrix(1,1); // NewMatrix is local
	auxLoss, _ := NewTensor(auxLossData, &TensorConfig{RequiresGrad: false, Name: "moe_aux_loss_field", Graph: graph}) // NewTensor is local

	return &MoELayer{ Config: config, Experts: experts, Router: router, AuxiliaryLoss: auxLoss }
}

// GetParameters method
func (ml *MoELayer) GetParameters() []*Tensor {
	params := ml.Router.GetParameters()
	for _, expert := range ml.Experts { params = append(params, expert.GetParameters()...) }
	return params
}

// Forward method for MoELayer
func (ml *MoELayer) Forward(inputTokens *Tensor, isTraining bool) (*Tensor, error) {
	if ml.Config.TopK <= 0 {
		fmt.Println("Warning: MoELayer TopK is <= 0, bypassing MoE layer.")
		ml.AuxiliaryLoss, _ = NewTensor(NewMatrixZeros(1,1), &TensorConfig{Graph: inputTokens.Graph, Name: "bypassed_aux_loss"}) // NewMatrixZeros is local
		return inputTokens, nil
	}

	graph := inputTokens.Graph
	numTotalTokens := inputTokens.Shape()[0]

	routerLogits, err := ml.Router.Forward(inputTokens); if err != nil { return nil, fmt.Errorf("router forward failed: %w", err) }

	if ml.Config.NoisyRouting && isTraining {
		fmt.Println("Note: Noisy routing is enabled but NewNormalTensor/Matrix is a TODO, skipping noise application.")
	}

	routerProbs, err := TensorSoftmax(routerLogits, -1); if err != nil { return nil, fmt.Errorf("router softmax failed: %w", err) } // TensorSoftmax is local

	topKRouterProbs, topKExpertIndices, err := TensorTopK(routerProbs, ml.Config.TopK, 1, true) // TensorTopK is local
	if err != nil {
		fmt.Printf("Error during TopK selection (K=%d): %v. MoE will be passthrough.\n", ml.Config.TopK, err)
		ml.AuxiliaryLoss, _ = NewTensor(NewMatrixZeros(1,1), &TensorConfig{Graph: graph, Name: "topk_err_aux_loss"})
		return inputTokens, nil
	}

	finalCombinedOutputConfig := &TensorConfig{Graph: graph, RequiresGrad: inputTokens.RequiresGrad, Name: "moe_final_output"}
	finalCombinedOutput, _ := NewZerosTensor(finalCombinedOutputConfig, inputTokens.Shape()...) // NewZerosTensor is local

	expertAssignmentMasksForLBLoss := make([]*Tensor, ml.Config.NumExperts)

	for kIdx := 0; kIdx < ml.Config.TopK; kIdx++ {
		currentExpertIndicesSlice, errSlice := TensorSlice(topKExpertIndices, []*SliceArg{{Start:0, End:numTotalTokens}, {Start:kIdx, End:kIdx+1}}, fmt.Sprintf("topk_idx_k%d", kIdx))
		if errSlice != nil {return nil, fmt.Errorf("slicing topKExpertIndices for k_idx %d: %w", kIdx, errSlice)}

		currentRouterProbsSlice, errSlice := TensorSlice(topKRouterProbs, []*SliceArg{{Start:0, End:numTotalTokens}, {Start:kIdx, End:kIdx+1}}, fmt.Sprintf("topk_prob_k%d", kIdx))
		if errSlice != nil {return nil, fmt.Errorf("slicing topKRouterProbs for k_idx %d: %w", kIdx, errSlice)}

		for e := 0; e < ml.Config.NumExperts; e++ {
			activeExpertMaskComparative, errEq := TensorEqualScalar(currentExpertIndicesSlice, float64(e)) // TensorEqualScalar is local
			if errEq != nil { fmt.Printf("Error TensorEqualScalar expert %d, k_idx %d: %v\n", e, kIdx, errEq); continue }

			activeExpertMaskF, errCast := TensorCast(activeExpertMaskComparative, Float64) // TensorCast & Float64 are local
			if errCast != nil { fmt.Printf("Error TensorCast expert %d, k_idx %d: %v\n", e, kIdx, errCast); continue }

			if kIdx == 0 {
				if expertAssignmentMasksForLBLoss[e] == nil {
					expertAssignmentMasksForLBLoss[e] = activeExpertMaskF
				}
			}

			currentExpertInput, errOp := Multiply(inputTokens, activeExpertMaskF) // Multiply is local
			if errOp != nil { fmt.Printf("Error Multiply expert input expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }

			expertOutput, errOp := ml.Experts[e].Forward(currentExpertInput)
			if errOp != nil { fmt.Printf("Error Expert %d Forward, k_idx %d: %v\n", e, kIdx, errOp); continue }

			gatingWeight, errOp := Multiply(currentRouterProbsSlice, activeExpertMaskF)
			if errOp != nil { fmt.Printf("Error Multiply gating weight expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }

			weightedExpertOutput, errOp := Multiply(expertOutput, gatingWeight)
			if errOp != nil { fmt.Printf("Error Multiply weighted output expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }

			tempFinalCombinedOutput, errOp := Add(finalCombinedOutput, weightedExpertOutput) // Add is local
			if errOp != nil { fmt.Printf("Error Add final output expert %d, k_idx %d: %v\n", e, kIdx, errOp); continue }
			finalCombinedOutput = tempFinalCombinedOutput
		}
	}

	auxLossConfig := &TensorConfig{Graph: graph, Name: "total_aux_loss_val", RequiresGrad: true}
	totalAuxLoss, _ := NewTensor(NewMatrixZeros(1,1), auxLossConfig)


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
			var P_i_tensor, f_i_tensor *Tensor; var errLBLoss error
			P_i_tensor, errLBLoss = TensorMean(routerProbs, 0, false)
			if errLBLoss != nil { fmt.Printf("Error calculating P_i for LB Loss: %v.\n", errLBLoss)
			} else {
				f_i_tensor = P_i_tensor

				if ml.Config.TopK == 1 && len(expertAssignmentMasksForLBLoss) == ml.Config.NumExperts {
					f_i_data_list := make([]float64, ml.Config.NumExperts)
					validFi := true
					for e := 0; e < ml.Config.NumExperts; e++ {
						if expertAssignmentMasksForLBLoss[e] == nil { validFi = false; break }
						sumMaskTensor, errSumMask := TensorMean(expertAssignmentMasksForLBLoss[e], -1, false)
						if errSumMask != nil { validFi = false; break }

						countTensorScaled, errScale := ScalarMultiply(sumMaskTensor, float64(expertAssignmentMasksForLBLoss[e].Shape()[0]))
						if errScale != nil {validFi = false; break }
						f_i_data_list[e] = countTensorScaled.Data()[0][0]
					}

					if validFi && numTotalTokens > 0 {
						f_i_matrix_data := NewMatrixFromSlice(f_i_data_list, 1, ml.Config.NumExperts) // NewMatrixFromSlice local
						f_i_tensor_unnormalized_cfg := &TensorConfig{Graph: graph, Name: "f_i_counts"}
						f_i_tensor_unnormalized, _ := NewTensor(f_i_matrix_data, f_i_tensor_unnormalized_cfg)
						f_i_tensor, _ = ScalarMultiply(f_i_tensor_unnormalized, 1.0/float64(numTotalTokens))
					} else { fmt.Println("Using P_i for f_i in Load Balancing Loss (issues with masks or zero tokens).") }
				} else if ml.Config.TopK > 1 {
					fmt.Println("Using P_i for f_i in Load Balancing Loss for TopK > 1 (more accurate f_i TODO).")
				}

				if f_i_tensor != nil && P_i_tensor != nil {
					products, errProd := Multiply(f_i_tensor, P_i_tensor)
					if errProd != nil { fmt.Printf("Error calculating products for LB Loss: %v.\n", errProd)
					} else {
						sumOverExperts, errSumProd := TensorMean(products, -1, false)
						if errSumProd != nil { fmt.Printf("Error summing products for LB Loss: %v.\n", errSumProd)
						} else {
							finalLBCoeff := ml.Config.LoadBalanceLossCoeff * float64(ml.Config.NumExperts*ml.Config.NumExperts)
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
		ml.AuxiliaryLoss, _ = NewTensor(NewMatrixZeros(1,1), &TensorConfig{Graph: graph, Name: "no_train_aux_loss"})
	}
	return finalCombinedOutput, nil
}
