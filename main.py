import numpy as np
from PIL import Image
from scipy import misc
import theano
import theano.tensor as T
import lasagne

import sys
import os
import time

def build_cnn(input_var=None): #returns a cnn: conv, max pool, two unit output
 network = lasagne.layers.InputLayer(shape=(None, 3, 128, 128),
            input_var=input_var)
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))

 network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))

 network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))

 network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))
 network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.sigmoid)
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))

 network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid)
 print("Layer: {}".format(lasagne.layers.get_output_shape(network)))
 return network

def format_input(inputs):
 new_list = list()
 for examples in inputs:
  new_list.append(np.expand_dims(examples, axis=0))
 new_array = np.vstack(new_list)
 return np.asarray(new_array)

def iterate_minibatch(data, batchsize, shuffle=False):
 for start_index in range(0, len(data) - batchsize + 1, batchsize):
  if(start_index % 1000 == 0):
   sys.stdout.write("{}...".format(start_index))
   sys.stdout.flush()
  inputs = np.asarray(data)[start_index: start_index+batchsize, 0]
  targets = np.asarray(data)[start_index: start_index+batchsize, 1]
  form_inputs = format_input(inputs)
  form_targets = format_input(targets)
  yield form_inputs.astype('float32'), form_targets.astype(bool)

def format_dataset():
 print("Formatting Dataset")
 data = list()
 for i in range(12500):
  cat = open("cat.{}.jpg".format(i), 'r+b')
  dog = open("dog.{}.jpg".format(i), 'r+b')
  cat_image = Image.open(cat)
  dog_image = Image.open(dog)
  cat_image_mod = cat_image.resize([128,128])
  dog_image_mod = dog_image.resize([128,128])
  cat_image_mod.save("cat.{}.jpg.modified".format(i), cat_image.format)
  dog_image_mod.save("dog.{}.jpg.modified".format(i), dog_image.format)
  cat_im = misc.imread("cat.{}.jpg.modified".format(i))
  dog_im = misc.imread("dog.{}.jpg.modified".format(i))
  cat_im_rolled = np.rollaxis(cat_im, 2, 0)
  dog_im_rolled = np.rollaxis(dog_im, 2, 0)
  data.append([cat_im_rolled, 0])
  data.append([dog_im_rolled, 1])
 return data #adjust this to change how many examples to set aside for testing


#main script
data = format_dataset()
print("Shuffling the Data...")
np.random.shuffle(data)
input_var = T.ftensor4('inputs') #theano variable supposed to be (f,f,f,f)
target_var = T.bmatrix('targets') #theano variable supposed to be
network = build_cnn(input_var)

print("Setting Theano Parameters...")
 #here are the theano training parameters
 # Create a loss expression for training
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
loss = lasagne.objectives.aggregate(loss, mode='mean')
 # Create update expressions for training, (SGD) 
params = lasagne.layers.get_all_params(network, trainable=True)
#updates = lasagne.updates.sgd(loss, params, learning_rate=0.01) #change learning rate of sgd here
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
 # Compile a function performing a training step on a mini-batch
train_fn = theano.function([input_var, target_var], loss, updates=updates)
 #here are the theano testing parameters
 # Create a loss expression for validation/testing. Different from training.
#predictions = theano.tensor.ge(test_prediction, 0.5)
#test_acc = T.mean(theano.tensor.eq(predictions, target_var), dtype=theano.config.floatX)
 # Compile a second function computing the validation loss and accuracy:

 #now we train the network
f = open('log.txt', 'w')
print("Starting Training...")
for epoch in range(50): #im hard coding epoch number for now
 train_err, train_batches = 0, 0
 start_time = time.time()
 print("Epoch {}".format(epoch + 1))
 f.write("Epoch {}\n".format(epoch + 1))
 for batch in iterate_minibatch(data, 200, shuffle=False):
  inputs, targets = batch
  train_err += train_fn(inputs, targets)
  train_batches += 1
 print("\ntraining loss:\t\t{:.6f}".format(train_err / train_batches))
 f.write("\ntraining loss:\t\t{:.6f}\n".format(train_err / train_batches))
 print("Final results in {}s".format(time.time() - start_time))
 if(epoch > 0):
  if(epoch % 10 == 0):
   print("Saving Parameters...")
   np.savez("model_weightse{}.npz".format(epoch), *lasagne.layers.get_all_param_values(network))
   

print("Saving Parameters...")
np.savez("model_weights.npz", *lasagne.layers.get_all_param_values(network))
