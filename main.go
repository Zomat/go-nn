package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/Zomat/go-nn/plot"
	"gonum.org/v1/plot/plotter"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

type Neuron struct {
	Weights       []float64
	Bias          float64
	Output        float64
	Delta         float64
	WeightsPoints []plotter.XYs
	PreviousWeightUpdates []float64
	PreviousBiasUpdate   float64
}

type Layer struct {
	Neurons         []Neuron
	MsePoints       plotter.XYs
	EpochMsePoints  plotter.XYs
	Momentum     	 float64
}

type Network struct {
	Layers       []Layer
	LearningRate float64
}

func NewNeuron(inputSize int) Neuron {
	weights := make([]float64, inputSize)
	for i := range weights {
		weights[i] = rand.NormFloat64()
	}
	return Neuron{
		Weights: weights,
		Bias:    0,
    WeightsPoints: make([]plotter.XYs, inputSize),
    PreviousWeightUpdates: make([]float64, inputSize),
	}
}

func NewLayer(neuronCount, inputSize int, momentum float64) Layer {
	neurons := make([]Neuron, neuronCount)
	for i := range neurons {
		neurons[i] = NewNeuron(inputSize)
	}
	return Layer{
		Neurons: neurons,
		Momentum: momentum,
	}
}

func NewNetwork(sizes []int, learningRate float64, momentum float64) Network {
	layers := make([]Layer, len(sizes)-1)
	for i := 0; i < len(sizes)-1; i++ {
		layers[i] = NewLayer(sizes[i+1], sizes[i], momentum)
	}
	return Network{
		Layers:       layers,
		LearningRate: learningRate,
	}
}

func (n *Neuron) Activate(inputs []float64) float64 {
	sum := n.Bias
	for i, weight := range n.Weights {
		sum += weight * inputs[i]
	}
	return sigmoid(sum)
}

func (l *Layer) Forward(inputs []float64, recordWeights bool) []float64 {
	outputs := make([]float64, len(l.Neurons))
	for i := range l.Neurons {
		neuron := &l.Neurons[i]
		neuron.Output = neuron.Activate(inputs)

		if recordWeights {
		  for wIdx := range neuron.Weights {
        neuron.WeightsPoints[wIdx] = append(neuron.WeightsPoints[wIdx], plotter.XY{
          X: float64(len(neuron.WeightsPoints[wIdx])),
          Y: neuron.Weights[wIdx],
        })
      }	
	  }
		outputs[i] = neuron.Output
	}
	return outputs
}

func (l *Layer) Backward(prevOutputs []float64, target []float64, nextLayer *Layer, isOutputLayer bool, learningRate float64) float64 {
	var totalLayerError float64
	for i := range l.Neurons {
		neuron := &l.Neurons[i]
		var err float64
		if isOutputLayer {
			err = target[i] - neuron.Output
		} else {
			for _, nextNeuron := range nextLayer.Neurons {
				err += nextNeuron.Delta * nextNeuron.Weights[i]
			}
		}

		totalLayerError += err * err
		neuron.Delta = err * sigmoidDerivative(neuron.Output)

		for j := range neuron.Weights {
			neuron.Weights[j] += learningRate * neuron.Delta * prevOutputs[j]
			change := learningRate*neuron.Delta*prevOutputs[j] + l.Momentum*neuron.PreviousWeightUpdates[j]
			neuron.Weights[j] += change
			neuron.PreviousWeightUpdates[j] = change

			biasChange := learningRate*neuron.Delta + l.Momentum*neuron.PreviousBiasUpdate
			neuron.Bias += biasChange
			neuron.PreviousBiasUpdate = biasChange
		}
		neuron.Bias += learningRate * neuron.Delta
	}

	l.MsePoints = append(l.MsePoints, plotter.XY{
		X: float64(len(l.MsePoints)),
		Y: totalLayerError / float64(len(l.Neurons)),
	})

	return totalLayerError
}

func (net *Network) Forward(input []float64, recordWeights bool) []float64 {
	out := input
	for i := range net.Layers {
		out = net.Layers[i].Forward(out, recordWeights)
	}
	return out
}

func (net *Network) Backward(inputs []float64, target []float64, recordWeights bool) {
	outputs := make([][]float64, len(net.Layers)+1)
	outputs[0] = inputs
	for i := range net.Layers {
		outputs[i+1] = net.Layers[i].Forward(outputs[i], recordWeights)
	}

	for i := len(net.Layers) - 1; i >= 0; i-- {
		isOutput := i == len(net.Layers)-1
		var nextLayer *Layer
		if !isOutput {
			nextLayer = &net.Layers[i+1]
		}
		var tgt []float64
		if isOutput {
			tgt = target
		}
		net.Layers[i].Backward(outputs[i], tgt, nextLayer, isOutput, net.LearningRate)
	}
}

func (net *Network) Train(inputs []float64, target []float64, recordWeights bool) {
	net.Backward(inputs, target, recordWeights)
}

func (net *Network) Predict(inputs []float64) []float64 {
	return net.Forward(inputs, false)
}

func main() {
	net := NewNetwork([]int{2, 2, 1}, 0.25, 0.9)

	var totalErrorPoints plotter.XYs
	var accPoints plotter.XYs

	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := []float64{0, 1, 1, 0}

	epochs := 2000
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		acc := 0

		for i := 0; i < len(inputs); i++ {
			recordWeights := i == 0
			net.Train(inputs[i], []float64{targets[i]}, recordWeights)

			pred := net.Predict(inputs[i])
			err := targets[i] - pred[0]
			totalError += err * err

			predictedClass := 0
			if pred[0] >= 0.5 {
				predictedClass = 1
			}
			if predictedClass != int(targets[i]) {
				acc += 1
			}
		}

		for _, layer := range net.Layers {
			layerError := 0.0
			for i := 0; i < len(inputs); i++ {
				output := net.Predict(inputs[i])
				err := targets[i] - output[0]
				layerError += err * err
			}
			layer.EpochMsePoints = append(layer.EpochMsePoints, plotter.XY{
				X: float64(epoch),
				Y: layerError / float64(len(inputs)),
			})
		}

		totalErrorPoints = append(totalErrorPoints, plotter.XY{X: float64(epoch), Y: totalError})
		accPoints = append(accPoints, plotter.XY{X: float64(epoch), Y: float64(acc) / float64(len(inputs))})

		if epoch%(epochs/100) == 0 {
			fmt.Printf("Epoch %d, Error: %.4f\n", epoch, totalError)
		}

		if totalError < 0.01 {
			fmt.Printf("Early stopping at epoch %d\n", epoch)
			break
		}
	}

	fmt.Println("\n== Test ==")
	for i := 0; i < len(inputs); i++ {
		out := net.Predict(inputs[i])
		binary := 0
		if out[0] > 0.5 {
			binary = 1
		}
		fmt.Printf("%v => %.4f => %d\n", inputs[i], out[0], binary)
	}

	var layerErrorPoints [][]plotter.XYs = make([][]plotter.XYs, 4)
	for layerIdx := range net.Layers {
		for sampleIdx := 0; sampleIdx < 4; sampleIdx++ {
			var filtered plotter.XYs
			count := 0
			for epoch := sampleIdx; epoch < len(net.Layers[layerIdx].MsePoints); epoch += 4 {
				filtered = append(filtered, plotter.XY{
					X: float64(count),
					Y: net.Layers[layerIdx].MsePoints[epoch].Y,
				})
				count++
			}
			layerErrorPoints[sampleIdx] = append(layerErrorPoints[sampleIdx], filtered)
		}
	}

	plot.SavePlot([]plotter.XYs{totalErrorPoints}, "MSE - dla całego zbioru danych", "Epoka", "MSE", []string{"MSE"}, "plots/total_error_plot.png")
	plot.SavePlot([]plotter.XYs{accPoints}, "Błąd klasyfikacji", "Epoka", "Błąd", []string{"Błąd"}, "plots/acc_plot.png")

	plot.SavePlot(layerErrorPoints[0], "MSE - przykład {0, 0}", "Epoka", "MSE", []string{"Warstwa 1", "Warstwa 2"}, "plots/layer_error_plot_1.png")
	plot.SavePlot(layerErrorPoints[1], "MSE - przykład {0, 1}", "Epoka", "MSE", []string{"Warstwa 1", "Warstwa 2"}, "plots/layer_error_plot_2.png")
	plot.SavePlot(layerErrorPoints[2], "MSE - przykład {1, 0}", "Epoka", "MSE", []string{"Warstwa 1", "Warstwa 2"}, "plots/layer_error_plot_3.png")
	plot.SavePlot(layerErrorPoints[3], "MSE - przykład {1, 1}", "Epoka", "MSE", []string{"Warstwa 1", "Warstwa 2"}, "plots/layer_error_plot_4.png")

	for layerIdx, layer := range net.Layers {
    for neuronIdx, neuron := range layer.Neurons {
      labels := []string{}
      for w := range neuron.WeightsPoints {
        labels = append(labels, fmt.Sprintf("Waga %d", w))
      }
      plot.SavePlot(
        neuron.WeightsPoints,
        fmt.Sprintf("Wagi neuronu %d w warstwie %d", neuronIdx+1, layerIdx+1),
        "Epoka", "Waga",
        labels,
        fmt.Sprintf("plots/layer_%d_neuron_%d_weights_plot.png", layerIdx+1, neuronIdx+1),
      )
    }
  }
}
