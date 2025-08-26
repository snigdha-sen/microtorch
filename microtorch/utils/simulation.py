import numpy as np


#add noise to signals
def add_noise(data, scale=0.02):
    data_real = data + np.random.normal(scale=scale, size=np.shape(data))
    data_imag = np.random.normal(scale=scale, size=np.shape(data))
    data_noisy = np.sqrt(data_real**2 + data_imag**2)

    return data_noisy
  
  
  