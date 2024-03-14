import numpy as np
from codebase.networks.components import conv_layer, bn_layer
from codebase.networks.layer import LeakyRelu

class ResidualBlock:
    def __init__(self, activation=LeakyRelu(), in_channels=16, out_channels=16, stride=1, id="res_block") -> None:
        self.id = id

        self.activation = activation

        self.conv1 = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_h=3,
            kernel_w=3,
            same=True,
            stride=1,
            shift=False,
            id=self.id + "_conv1"
        )

        self.bn1 = bn_layer(
            neural_num=out_channels,
            moving_rate=0.1,
            id=self.id + "_bn1"
        )

        
        self.conv2 = conv_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_h=3,
            kernel_w=3,
            same=True,
            stride=1,
            shift=False,
            id=self.id + "_conv2"
        )

        self.bn2 = bn_layer(
            neural_num=out_channels,
            moving_rate=0.1,
            id=self.id + "_bn2"
        )

        self.cache = {}



    def get_params(self, params):
        self.conv1.get_params(params)
        self.bn1.get_params(params)
        self.conv2.get_params(params)
        self.bn2.get_params(params)

    def set_params(self, params):
        self.conv1.set_params(params)
        self.bn1.set_params(params)
        self.conv2.set_params(params)
        self.bn2.set_params(params)  
    

    def forward(self, x: np.ndarray):
        x_init = x.copy()
        x1 = x
        x1 = self.conv1.forward(x1)
        x1 = self.bn1.forward(x1)
        x1, relu0_cache = self.activation.forward(x1)

        x1 = self.conv2.forward(x1)
        x1 = self.bn2.forward(x1)

        x1 = x1 + x_init

        x1, relu1_cache = self.activation.forward(x1)
        out = x1 

        self.relu0_cache = relu0_cache
        self.relu1_cache = relu1_cache
        
        return out

    def backward(self, dy: np.ndarray, grads):
        dy = self.activation.backward(dy, self.relu1_cache)
        dx_init = dy.copy()
        dy = self.bn2.backward(dy, grads)
        dy = self.conv2.backward(dy, grads)
        dy = self.activation.backward(dy, self.relu0_cache)
        dy = self.bn1.backward(dy, grads)
        dy = self.conv1.backward(dy, grads)
        dy += dx_init

        return dy
