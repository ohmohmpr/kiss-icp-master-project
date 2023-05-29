
import numpy as np

my_dict_back = np.load('thisdict.npy', allow_pickle='TRUE').item()


print(my_dict_back[252].shape)