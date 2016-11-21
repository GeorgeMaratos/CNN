import numpy as np
import pickle

f1 = open('animal_data.pkl', 'rb')
final_list = pickle.load(f1)
final_array = np.asarray(final_list)
print(final_array[0,0].shape)
