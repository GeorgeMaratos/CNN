import numpy as np
from scipy import misc

animal_list = list()
for i in range(12500):
 im = misc.imread("cat_scaled.{}.jpg".format(i))
 animal_list.append(im)
 im = misc.imread("dog_scaled.{}.jpg".format(i))
 animal_list.append(im)

animal_list = np.expand_dims(animal_list, axis=0)
animal_data = np.dstack(animal_list)
animal_data = np.rollaxis(animal_data, 3, 1)
print(animal_data.shape)
f = open('animal_data.npy', 'wb')
np.save(f, animal_data)
