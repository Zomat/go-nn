package main

import (
  "fmt"
  "math"
  "math/rand"
)

func sigmoid(x float64) float64 {
  return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
  return x * (1 - x)
}

type Neuron struct {
  Weights []float64
  Bias    float64
  Output  float64
  Delta   float64
}

type Layer struct {
  Neurons []Neuron
}

type Network struct {
  InputSize   int
  HiddenLayer Layer
  OutputLayer Layer
  LearningRate float64
}

func NewNeuron(inputSize int) Neuron {
  weights := make([]float64, inputSize)
  for i := range weights {
    weights[i] = rand.NormFloat64() * 0.5
  }
  return Neuron{
    Weights: weights,
    Bias:    rand.NormFloat64() * 0.5,
  }
}

func NewLayer(neuronCount, inputSize int) Layer {
  neurons := make([]Neuron, neuronCount)
  for i := range neurons {
    neurons[i] = NewNeuron(inputSize)
  }
  return Layer{
    Neurons: neurons,
  }
}

func NewNetwork(inputSize, hiddenLayerSize, outputLayerSize int, learningRate float64) Network {
  hiddenLayer := NewLayer(hiddenLayerSize, inputSize)
  outputLayer := NewLayer(outputLayerSize, hiddenLayerSize)

  return Network{
    InputSize:   inputSize,
    HiddenLayer: hiddenLayer,
    OutputLayer: outputLayer,
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

func (l *Layer) Forward(inputs []float64) []float64 {
  outputs := make([]float64, len(l.Neurons))
  for i := range l.Neurons {
    neuron := &l.Neurons[i]
    neuron.Output = neuron.Activate(inputs)
    outputs[i] = neuron.Output
  }
  return outputs
}

func (net *Network) Train(inputs []float64, target float64) {
  hiddenOutputs := net.HiddenLayer.Forward(inputs)
  finalOutputs := net.OutputLayer.Forward(hiddenOutputs)

  for i := range net.OutputLayer.Neurons {
    neuron := &net.OutputLayer.Neurons[i]
    err := target - finalOutputs[i]
    neuron.Delta = err * sigmoidDerivative(finalOutputs[i])

    for j := range neuron.Weights {
      neuron.Weights[j] += net.LearningRate * neuron.Delta * hiddenOutputs[j]
    }
    neuron.Bias += net.LearningRate * neuron.Delta
  }

  for i := range net.HiddenLayer.Neurons {
    neuron := &net.HiddenLayer.Neurons[i]
    errSum := 0.0

    for _, outNeuron := range net.OutputLayer.Neurons {
      errSum += outNeuron.Delta * outNeuron.Weights[i]
    }

    neuron.Delta = errSum * sigmoidDerivative(neuron.Output)

    for j := range neuron.Weights {
      neuron.Weights[j] += net.LearningRate * neuron.Delta * inputs[j]
    }
    neuron.Bias += net.LearningRate * neuron.Delta
  }
}

func (net *Network) Predict(inputs []float64) []float64 {
  hiddenOutputs := net.HiddenLayer.Forward(inputs)
  return net.OutputLayer.Forward(hiddenOutputs)
}

func main() {
  net := NewNetwork(2, 2, 1, 0.5)

	// XOR dane
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := []float64{
		0,
		1,
		1,
		0,
	}

	for epoch := 0; epoch < 100000; epoch++ {
		totalError := 0.0
		for i := 0; i < len(inputs); i++ {
			net.Train(inputs[i], targets[i])
			pred := net.Predict(inputs[i])
			err := targets[i] - pred[0]
			totalError += err * err
		}
		if epoch%10000 == 0 {
			fmt.Printf("Epoch %d, Error: %.4f\n", epoch, totalError)
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
}
