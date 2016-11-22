import numpy as np
from scipy import misc
import pickle

#this code will reads all images and formats them to (25000,2) where each object on left is (3,128,128)
print("Preparing the Data...")
animal_list = list()
for i in range(12500):
 im = misc.imread("cat_scaled.{}.jpg".format(i))
 animal_list.append([np.rollaxis(np.asarray(im),2,0), np.float32(1)])
 im = misc.imread("dog_scaled.{}.jpg".format(i))
 animal_list.append([np.rollaxis(np.asarray(im),2,0), np.float32(0)])


print("Writing to file...")
f = open('animal_list.pkl', 'wb')
pickle.dump(animal_list, f, protocol=2)
