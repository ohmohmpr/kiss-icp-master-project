
import numpy as np

my_dict_back = np.load('thisdict.npy', allow_pickle='TRUE').item()

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

max = 0 
for i in range(len(my_dict_back)):
    if len(my_dict_back[i]) > max:
        max = len(my_dict_back[i])
print(max)