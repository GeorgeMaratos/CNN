import numpy as np
from scipy import misc
import pickle

print("Creating list...")
animal_list = list()
for i in range(12500):
 im = misc.imread("cat_scaled.{}.jpg".format(i))
 animal_list.append(im)
 im = misc.imread("dog_scaled.{}.jpg".format(i))
 animal_list.append(im)

animal_list = np.expand_dims(animal_list, axis=0)
animal_data = np.dstack(animal_list)
animal_data = np.rollaxis(animal_data, 3, 1)
final_list = list()
print("Appending label...")
for i in range(0,25000,2):
 final_list.append([animal_data[i,:,:,:], 1])
 final_list.append([animal_data[i+1,:,:,:], 0])

print("Writing to animal_data.pkl...")
f = open('animal_data.pkl', 'wb')
pickle.dump(final_list, f, protocol=2)
