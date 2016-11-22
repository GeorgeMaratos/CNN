import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle


def build_cnn(input_var=None): #returns a cnn: conv, max pool, two unit output
 print("Building Network...")
 #input layer
 network = lasagne.layers.InputLayer(shape=(None, 3, 128, 128), input_var=input_var)
 #convolution and max pooling layer
 network = lasagne.layers.Conv2DLayer(
        network, num_filters=64, filter_size=(5,5),
        nonlinearity=lasagne.nonlinearities.rectify)
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
 #2 unit softmax output layer
 network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)
 return network

def format_input(inputs):
 new_list = list()
 for examples in inputs:
  new_list.append(np.expand_dims(examples, axis=0))
 new_array = np.vstack(new_list)
 return np.asarray(new_array)

def iterate_minibatch(data, batchsize, shuffle=False):
 for start_index in range(0, len(data) - batchsize + 1, batchsize):
  inputs = np.asarray(data)[start_index: start_index+batchsize, 0]
  targets = np.asarray(data)[start_index: start_index+batchsize, 1]
  form_inputs = format_input(inputs)
  yield form_inputs.astype('float32'), targets.astype(bool)

def load_data(): #This will return a numpy array of tuples (25000,2) 
 print("Loading Data...")
 f1 = open('animal_array.npy', 'rb')
 animal_array = np.load(f1)
 return animal_array[20000:], animal_array[-5001:] 

#MAIN SCRIPT
data, test_data = load_data()
input_var = T.ftensor4('inputs') #theano variable supposed to be (f,f,f,f)
target_var = T.lvector('targets') #theano variable supposed to be (f,) it is possible this should be a float
network = build_cnn(input_var)
print("Setting Theano Parameters...")
 #here are the theano training parameters
 # Create a loss expression for training
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
 # Create update expressions for training, (SGD) 
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.sgd(loss, params, learning_rate=0.001) #change learning rate of sgd here
 # Compile a function performing a training step on a mini-batch
train_fn = theano.function([input_var, target_var], loss, updates=updates)
 #here are the theano testing parameters
 # Create a loss expression for validation/testing. Different from training.
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
 #theano function for testing accuracy
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
 # Compile a second function computing the validation loss and accuracy:
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

 #now we train the network
print("Starting Training...")
for epoch in range(50): #im hard coding epoch number for now
 train_err, train_batches = 0, 0
 for batch in iterate_minibatch(data, 200, shuffle=False):
  inputs, targets = batch
  train_err += train_fn(inputs, targets)
  train_batches += 1
 print("Epoch {} of {}".format(epoch + 1, 500))
 print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
 test_err, test_acc, test_batches = 0, 0, 0
 for batch in iterate_minibatch(test_data, 200, shuffle=False):
  inputs, targets = batch
  err, acc = val_fn(inputs, targets)
  test_err += err
  test_acc += acc
  test_batches += 1
 print("Final results:")
 print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
 print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

print("Saving Parameters...")
np.savez('model.npz', *lasagne.layers.get_all_param_values(network))

