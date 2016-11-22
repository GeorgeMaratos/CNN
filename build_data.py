import numpy as np
from scipy import misc
import pickle

print("Preparing the Data")
animal_list = list()

for i in range(12500):
 im = misc.imread("cat_scaled.{}.jpg".format(i))
 animal_list.append(im)
 im = misc.imread("dog_scaled.{}.jpg".format(i))
 animal_list.append(im)

animal_data = np.asarray(animal_list)
animal_data = np.rollaxis(animal_data, 3, 1)
print(animal_data.shape)
