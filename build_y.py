import numpy as np

y_train = list()
for i in range(12500):
 y_train.append(1)
 y_train.append(-1)
y_train_arr = np.asarray(y_train)
print(y_train_arr.shape)
f = open('y_train.npy', 'wb')
np.save(f,y_train_arr)
