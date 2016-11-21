import numpy as np
import pickle

def load_data():
 f1 = open('animal_data.pkl', 'rb')
 final_list = pickle.load(f1)
 final_array = np.asarray(final_list)
 print(final_array[0,0].shape)

 new_list = list()
 for i in range(25000):
  new_list.append(np.expand_dims(final_array[i,0], axis=0))

 new_array = np.asarray(new_list)
 x_train = np.vstack(new_array)
 print(x_train.shape)

 return x_train, final_array[:,1]
 
load_data()
