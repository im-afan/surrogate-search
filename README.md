# UCSB Summer Research Academies -- Neuromorphic Computing (Track 12) 
## Trainable Stochastic Surrogate Functions for Direct Spiking Neural Network Training
### Abstract 
  Spiking neural networks (SNNs), which are inspired by biological systems, have emerged as a solution to
address the energy-intensive computations associated with artificial neural networks (ANNs). Current approaches
to training SNNs include converting ANNs to SNNs and adapting backpropagation methods. However, neuron
spikes pose challenges for gradient-based optimization due to the non-differentiable step function. Surrogate
gradient descent addresses these challenges by using surrogate functions to approximate step function gradients.
However, constant surrogate functions are inflexible and limit SNNs from achieving optimal performance
comparable to ANNs. In this paper, we introduce a novel approach using an adaptable surrogate function for SNN
training. This is done by sampling the surrogate width from a normal distribution with trainable mean and standard
deviation. A policy gradient method inspired by reinforcement learning is used to update the trainable parameters.
We compare our method with existing methods on the CIFAR-10 dataset and utilize the VGG-16 architecture. We
find that our method achieves a slightly better accuracy than vanilla surrogate gradient descent. However, our
model does not match the accuracy of state-of-the-art methods, possibly due to several factors. Still, our method
shows theoretical promise to be improved upon and enhance future SNN training.
