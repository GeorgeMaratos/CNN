import numpy as np
import theano
import theano.tensor as T
import lasagne

def build_cnn(input_var=None): #returns a cnn: conv, max pool, two unit output
 print("Building Network...")
 #input layer
 input_layer = lasagne.layers.InputLayer(shape=(None, 3, 128, 128), input_var=input_var)
 #convolution and max pooling layer
 network = lasagne.layers.Conv2DLayer(
        input_layer, num_filters=64, filter_size=(5,5),
        nonlinearity=lasagne.nonlinearities.rectify)
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 #2 unit softmax output layer
 network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)
 return input_layer, network

input_layer, network = build_cnn()
input_variable = T.tensor4('inputs')

with np.load('model.npz') as f:
 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
 lasagne.layers.set_all_param_values(network, param_values)

predict = theano.function([input_layer.input_var], lasagne.layers.get_output(network))