import numpy as np
from scipy import misc
import pickle

print("Preparing the Data")
animal_list = list()
for i in range(12500):
 im = misc.imread("cat_scaled.{}.jpg".format(i))
 animal_list.append([np.rollaxis(np.asarray(im),2,0), np.float32(1)])
 im = misc.imread("dog_scaled.{}.jpg".format(i))
 animal_list.append([np.rollaxis(np.asarray(im),2,0), np.float32(1)])
print(np.asarray(animal_list)[0][0].shape)
