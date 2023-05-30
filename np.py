
import numpy as np

my_dict_back = np.load('thisdict.npy', allow_pickle='TRUE').item()

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(my_dict_back[0])