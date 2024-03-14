import numpy as np
import os
import pickle
from codebase.networks.components import conv_layer, bn_layer
from codebase.networks.layer import affine_forward, affine_backward, Sigmoid, Relu
from codebase.networks.base_networks import Network
from codebase.networks.resblock import ResidualBlock

class ClassificationNet(Network):
    """
    A fully-connected classification neural network with configurable 
    activation function, number of layers, number of classes, hidden size and
    regularization strength. 
    """

    def __init__(self, activation=Sigmoid(), num_layer=2,
                 input_size=3 * 32 * 32, hidden_size=100,
                 std=1e-3, num_classes=10, reg=0, **kwargs):
        """
        :param activation: choice of activation function. It should implement
            a forward() and a backward() method.
        :param num_layer: integer, number of layers. 
        :param input_size: integer, the dimension D of the input data.
        :param hidden_size: integer, the number of neurons H in the hidden layer.
        :param std: float, standard deviation used for weight initialization.
        :param num_classes: integer, number of classes.
        :param reg: float, regularization strength.
        """
        super(ClassificationNet, self).__init__("cifar10_classification_net")

        self.activation = activation
        self.reg_strength = reg

        self.cache = None

        self.memory = 0
        self.memory_forward = 0
        self.memory_backward = 0
        self.num_operation = 0

        # Initialize random gaussian weights for all layers and zero bias
        self.num_layer = num_layer
        self.params = {'W1': std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size)}

        for i in range(num_layer - 2):
            self.params['W' + str(i + 2)] = std * np.random.randn(hidden_size,
                                                                  hidden_size)
            self.params['b' + str(i + 2)] = np.zeros(hidden_size)

        self.params['W' + str(num_layer)] = std * np.random.randn(hidden_size,
                                                                  num_classes)
        self.params['b' + str(num_layer)] = np.zeros(num_classes)

        self.grads = {}
        self.reg = {}
        for i in range(num_layer):
            self.grads['W' + str(i + 1)] = 0.0
            self.grads['b' + str(i + 1)] = 0.0

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: Input data of shape N x D. Each X[i] is a training sample.
        :return: Predicted value for the data in X, shape N x 1
                 1-dimensional array of length N with the classification scores.
        """

        self.cache = {}
        self.reg = {}
        X = X.reshape(X.shape[0], -1)
        # Unpack variables from the params dictionary
        for i in range(self.num_layer - 1):
            W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]

            # Forward i_th layer
            X, cache_affine = affine_forward(X, W, b)
            self.cache["affine" + str(i + 1)] = cache_affine

            # Activation function
            X, cache_sigmoid = self.activation.forward(X)
            self.cache["sigmoid" + str(i + 1)] = cache_sigmoid

            # Store the reg for the current W
            self.reg['W' + str(i + 1)] = np.sum(W ** 2) * self.reg_strength

        # last layer contains no activation functions
        W, b = self.params['W' + str(self.num_layer)],\
               self.params['b' + str(self.num_layer)]
        y, cache_affine = affine_forward(X, W, b)
        self.cache["affine" + str(self.num_layer)] = cache_affine
        self.reg['W' + str(self.num_layer)] = np.sum(W ** 2) * self.reg_strength

        return y

    def backward(self, dy):
        """
        Performs the backward pass of the model.

        :param dy: N x 1 array. The gradient wrt the output of the network.
        :return: Gradients of the model output wrt the model weights
        """

        # Note that last layer has no activation
        cache_affine = self.cache['affine' + str(self.num_layer)]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads['W' + str(self.num_layer)] = \
            dW + 2 * self.reg_strength * self.params['W' + str(self.num_layer)]
        self.grads['b' + str(self.num_layer)] = db

        # The rest sandwich layers
        for i in range(self.num_layer - 2, -1, -1):
            # Unpack cache
            cache_sigmoid = self.cache['sigmoid' + str(i + 1)]
            cache_affine = self.cache['affine' + str(i + 1)]

            # Activation backward
            dh = self.activation.backward(dh, cache_sigmoid)

            # Affine backward
            dh, dW, db = affine_backward(dh, cache_affine)

            # Refresh the gradients
            self.grads['W' + str(i + 1)] = dW + 2 * self.reg_strength * \
                                           self.params['W' + str(i + 1)]
            self.grads['b' + str(i + 1)] = db

        return self.grads

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(directory + '/' + self.model_name + '.p', 'wb'))

    def get_dataset_prediction(self, loader):
        scores = []
        labels = []
        
        for batch in loader:
            X = batch['image']
            y = batch['label']
            score = self.forward(X)
            scores.append(score)
            labels.append(y)
            
        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()

        return labels, preds, acc


class BasicResNet(ClassificationNet):
    conv1: conv_layer

    def __init__(self, activation=Sigmoid(), dimsImg=[3, 32, 32],
                 std=1e-3, num_classes=10, reg=0, is_train=True, **kwargs):

        super(BasicResNet, self).__init__("cifar10_classification_net")

        self.params = {}
        self.grads = {}
        self.cache = {}
        
        self.reg_strength = reg
        self.activation = activation
        self.num_operation = 0
        self.is_train = is_train

        # PRE 
        self.conv0 = conv_layer(
            in_channels=3,
            out_channels=16,
            kernel_h=3,
            kernel_w=3,
            same=True,
            stride=2,
            shift=False,
            id="conv0"
        )
        self.conv0.get_params(self.params)
        self.bn0 = bn_layer(
            neural_num=16,
            moving_rate=0.1,
            id="bn0"
        )
        self.bn0.get_params(self.params)

        self.resblock = ResidualBlock(
                in_channels=16,
                out_channels=16,
                id="resblock"
        )
        self.resblock.get_params(self.params)

        self.relu = Relu()

        nDimsFlattened = int(16 * dimsImg[1] * dimsImg[2] / 4)

        self.params["W_fc"] = std * np.random.randn(nDimsFlattened, num_classes)
        self.params["b_fc"] = np.zeros(num_classes)

        if is_train:
            self.train()
        else:
            self.eval()

    def train(self):
        self.return_grad = True
    
    def eval(self):
        self.return_grad = False

    def forward(self, X):
        """
        X is assumed to be in shape
        (nBatches, nChannels, height, width)
        """

        if len(X.shape) == 2:
            images = X.reshape([X.shape[0], 32,  32,  3])
            reshape_order = 'F'
            nImgs = images.shape[0]
            nW = images.shape[1]
            nH = images.shape[2]
            nChannels = images.shape[3]
            X = images.reshape((nImgs, nChannels, nW, nH), order=reshape_order)
        out = None
        self.reg = {}

        #### PRE-CONVOLUTION
        self.conv0.set_params(self.params)
        out = self.conv0.forward(X)

        self.bn0.set_params(self.params)
        out = self.bn0.forward(out)

        out, relu0_cache = self.relu.forward(out)
        self.cache["relu0"] = relu0_cache

        #### RESIDUAL-BLOCK
        self.resblock.set_params(self.params)
        out = self.resblock.forward(out)

        reshaped_X = out.reshape(out.shape[0], -1)
        W, b = self.params["W_fc"], self.params["b_fc"]

        out, cache_affine = affine_forward(reshaped_X, W, b)
        self.cache["fc_affine"] = cache_affine

        return out

    def backward(self, dy):
        cache_affine = self.cache["fc_affine"]
        dh, dW, db = affine_backward(dy, cache_affine)
        self.grads["W_fc"] = dW
        self.grads["b_fc"] = db 
        dh  = dh.reshape((dh.shape[0], 16, 16, 16))
        dh = self.resblock.backward(dh, self.grads)
        cache_relu0 = self.cache["relu0"]
        dh = self.relu.backward(dh, cache_relu0)
        dh = self.bn0.backward(dh, self.grads)
        dx = self.conv0.backward(dh, self.grads)
        return self.grads

"""
(Epoch 1 / 5) train loss: 2.308361; val loss: 2.307882
(Epoch 2 / 5) train loss: 1.721183; val loss: 1.616421
(Epoch 3 / 5) train loss: 1.339638; val loss: 1.413771
(Epoch 4 / 5) train loss: 1.187380; val loss: 1.376826
(Epoch 5 / 5) train loss: 1.027868; val loss: 1.284143
Training accuray: 0.70189
Validation accuray: 0.55849
"""