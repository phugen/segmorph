"""layer {
  type: "Python"
  bottom: "in_cont"
  bottom: "in_binary"
  bottom: "in_cont"
  top: "out1"
  python_param: {
    module: "folder.my_layer_module_name"
    layer: "my_layer_class_name"
    param_str: "some params"
  }
  propagate_down: true
  propagate_down: false
}
Then you can test it's backward() gradients like this:
"""

import numpy as np
from test_gradient_for_python_layer import test_gradient_for_python_layer

# set the inputs
labels = np.array((1, 1, 2, 0)).reshape((1, 2, 2))
weights = np.array((0.25, 0.25, 1, 1)).reshape((1, 2, 2))

# Data: Batchsize=1 x Channels=3 x Height=2 x Width=2 x #Probablities=4
# Labels: Batchsize=1 x Height=2 x Width=2
# Weights: Batchsize=1 x Height=2 x Width=2
input_names_and_values = [('data', np.random.randn(1,3,2,2)), ('labels', labels), ('weights', weights)]
output_names = ['loss']
py_module = 'SoftmaxWithLossWeighted'
py_layer = 'SoftmaxWithLossWeighted'
propagate_down = [True, False, False]

# call the test
test_gradient_for_python_layer(input_names_and_values, output_names, py_module, py_layer, None, propagate_down)
