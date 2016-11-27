from __future__ import print_function
import numpy as np
from scipy import misc
from PIL import Image
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
        nonlinearity=lasagne.nonlinearities.sigmoid)
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3))
 #2 unit softmax output layer
 network = lasagne.layers.DenseLayer(
	network,
        num_units=1,
        nonlinearity=lasagne.nonlinearities.sigmoid)
 return input_layer, network

input_layer, network = build_cnn()
print("Loading Parameters..")
with np.load('model_weights.npz') as f:
 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
 lasagne.layers.set_all_param_values(network, param_values)
print("Building Predictor")
predict = theano.function([input_layer.input_var], lasagne.layers.get_output(network, deterministic=True))
print("Writing to file")
f = open('results.txt', 'w')
f.write("id,label\n")
for i in range(1,12501):
 f2 = open("{}.jpg".format(i))
 im = Image.open(f2)
 im_modified = im.resize([128,128])
 im_modified.save("{}.jpg.modified".format(i), im.format)
 im_modified_2 = misc.imread("{}.jpg.modified".format(i))
 im_modified_rolled = np.rollaxis(im_modified_2, 2, 0)
 im_final = np.expand_dims(im_modified_rolled, axis=0)
 f.write("{},{}\n".format(i, predict(im_final)[0][0]))
