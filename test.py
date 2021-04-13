import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# This is a script to test python using matlab files
# TestPicture_Z.mat
data = sio.loadmat("TestPicture_Z.mat")
print(data.keys())