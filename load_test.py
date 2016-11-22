import numpy as np
import pickle

f1 = open('animal_data.pkl', 'rb')
final_list = pickle.load(f1)
final_array = np.asarray(final_list)
x_train = final_array[:2000,0]
y_train = final_array[:2000,1]
print(x_train.shape)
