import numpy as np

f1 = open('x_train.npy', 'rb')
f2 = open('y_train.npy', 'rb')

x_train = np.load(f1)
y_train = np.load(f2)

print(x_train.shape)
print(y_train.shape)
