#main script

import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle

def load_data():
 print("Loading Data...")
 f1 = open('animal_data.pkl', 'rb')
 final_list = pickle.load(f1)
 final_array = np.asarray(final_list)
 return final_array

def build_cnn(input_var=None):

 print("Building Network...")
 #input layer
 network = lasagne.layers.InputLayer(shape=(None, 3, 128, 128), input_var=input_var)

 #convolution and max pooling layer
 network = lasagne.layers.Conv2DLayer(
        network, num_filters=32, filter_size=(3,3),
        nonlinearity=lasagne.nonlinearities.rectify)
 network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

 #2 unit softmax output layer
 network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

 return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs = 500): 

 x_train, y_train, x_test, y_test = load_data()
 input_var = T.ftensor4('inputs')
 target_var = T.lvector('targets')
 network = build_cnn(input_var) 

 # Create a loss expression for training
 prediction = lasagne.layers.get_output(network)
 loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
 loss = loss.mean()

 # Create update expressions for training,
 # (SGD) 
 params = lasagne.layers.get_all_params(network, trainable=True)
 updates = lasagne.updates.sgd(
        loss, params, learning_rate=0.05)

 # Create a loss expression for validation/testing. Different from training.
 test_prediction = lasagne.layers.get_output(network, deterministic=True)
 test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
 test_loss = test_loss.mean()

 #for test accuracy
 test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
         dtype=theano.config.floatX)

 # Compile a function performing a training step on a mini-batch
 train_fn = theano.function([input_var, target_var], loss, updates=updates)

 # Compile a second function computing the validation loss and accuracy:
 val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

 #my function to determine how predictions are being made
 pred = theano.function([input_var], [test_prediction])

 #start training
 print("Starting training...")

 for epoch in range(num_epochs):
  train_err = 0
  train_batches = 0

  for batch in iterate_minibatches(x_train, y_train, 500, shuffle=True):
   inputs, targets = batch
   train_err += train_fn(inputs, targets)
   train_batches += 1

  # Then we print the results for this epoch:
  print("Epoch {} of {}".format(epoch + 1, num_epochs))
  print(" training loss:\t\t{:.6f}\n".format(train_err / train_batches))

  test_err = 0
  test_acc = 0
  test_batches = 0
  for batch in iterate_minibatches(x_test, y_test, 500, shuffle=False):
   inputs, targets = batch
   err, acc = val_fn(inputs, targets)
   test_err += err
   test_acc += acc
   test_batches += 1
  print("Final results:")
  print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
  print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

main() 
