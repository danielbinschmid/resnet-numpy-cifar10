from codebase.networks.components import conv_layer, bn_layer
from codebase.networks.layer import Relu
from codebase.networks.resblock import ResidualBlock

class ResNetBlock:
    def __init__(self, in_channels=16, out_channels=32, id="block") -> None:
        assert in_channels * 2 == out_channels

        self.id = id
        self.conv_id = self.id + "_upscaleChannels-downscaleResol_conv"
        self.bn_id = self.conv_id + "-batchnorm"

        self.conv = conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_h=3,
            kernel_w=3,
            same=True,
            stride=2,
            shift=True,
            id=self.conv_id
        )

        self.bn = bn_layer(
            neural_num=out_channels,
            moving_rate=0.1,
            id=self.bn_id
        )
        
        self.activation = Relu()

        self.resblock_id = self.id + "_residblock"
        self.resblock = ResidualBlock(
            activation=Relu(),
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            id=self.resblock_id
        )

    def get_params(self, params):
        self.conv.get_params(params)
        self.resblock.get_params(params)
        self.bn.get_params(params)

    def set_params(self, params):
        self.bn.set_params(params)
        self.conv.set_params(params)
        self.resblock.set_params(params)

    def forward(self, x):
        
        out = self.conv.forward(x)
        out = self.bn.forward(out)

        out, cache = self.activation.forward(out)
        self.act_cache = cache
        out = self.resblock.forward(out) 

        return out


    def backward(self, dy, grads):
        dh = self.resblock.backward(dy, grads)

        dh = self.activation.backward(dh, self.act_cache)

        dh = self.bn.backward(dh, grads)

        dh = self.conv.backward(dh, grads)
        
        return dh 

