from scipy import optimize

class Teacher():

    def train(self, student, inputs, outputs):
        self._costs = []
        self._input = inputs
        self._output = outputs
        weights = student.get_weights()
        optimize_settings = {
            'jac': True,
            'method': 'BFGS',
            'args': (inputs, outputs),
            'options': {
                'maxiter': 2000,
                'disp': True
            },
            'callback': student.set_weights
        }
        result = optimize.minimize(student.optimize_interfacer,
                                   weights,
                                   **optimize_settings)
        student.set_weights(result.x)