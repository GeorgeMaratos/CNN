import sys
import numpy as np
from PIL import Image
from scipy import misc
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

filename = sys.argv[1]
im_file = open(filename, 'r+b')
im = Image.open(im_file)
im_modified = im.resize([128,128])
im_modified.save("{}.modified".format(filename), im.format)
im_modified_2 = misc.imread("{}.modified".format(filename))
im_modified_rolled = np.rollaxis(im_modified_2, 2, 0)
im_final = np.expand_dims(im_modified_rolled, axis=0)
input_layer, network = build_cnn()
print("Loading Parameters..")
with np.load('model.npz') as f:
 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
 lasagne.layers.set_all_param_values(network, param_values)
print("Building Predictor...")
predict = theano.function([input_layer.input_var], lasagne.layers.get_output(network))
print(predict(im_final))
