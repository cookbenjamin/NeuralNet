# neuralNet

A super simple artificial neural network in python that you can use to approximate any function.

## Requirements

The code was written with the intent of being interpreted as Python 3.

This neural network depends upon both NumPy and SciPy. Both belong to the scipy stack.
You can install them by following the [scipy stack installation instructions](http://scipy.org/install.html#individual-packages).

## Usage

```
Import:
>>> from neuralNet import NeuralNet

>>> nn = NeuralNet()

>>> x = [[3, 5]]
>>> y = [[.75]]

Train:
>>> nn.train(x, y)

Predict:
>>> nn.predict(x)
[[.750000000]]
```
