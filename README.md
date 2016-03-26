# neuralNet

A super simple artificial neural network in python that you can use to approximate any function.

## Requirements

This neural network depends upon both numpy and scipy. Both belong to the scipy stack.
You can install them by following the [scipy stack installation instructions](http://scipy.org/install.html#individual-packages).

## Usage

```
Import:
>>> from neuralNet import NeuralNet
>>> import numpy as np

>>> x = np.array([[3, 5]])
>>> y = np.array([[.75]])

>>> nn = NeuralNet()

Train:
>>> nn.train(x, y)

Predict:
>>> nn.predict(x)
[[.750000000]]
```
