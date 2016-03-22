# neuralNet

A super simple artificial neural network in python

## Usage

```
Import:
>>> from neuralNet import NeuralNet
>>> import numpy as np

>>> x = np.array([3, 5])
>>> y = np.array([.75])

>>> nn = NeuralNet()

Train:
>>> nn.train(x, y)

Predict:
>>> nn.predict(x)
.75
```

### DON'T EXPECT THIS TO WORK