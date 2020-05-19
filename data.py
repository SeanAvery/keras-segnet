from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

class Dataset(keras.utils.Sequence):
  """ helper to load comma 10k data set """
  
  def __init__(self, batch_size, img_size, input_imgs, output_imgs):
    self.batch_size = batch_size
    self.img_size = img_size
    self.input_imgs = input_imgs
    self.output_imgs = output_imgs
    
  def __len__(self):
    return len(self.output_imgs) // self.batch_size
  
  def __getitem__(self, idx):
    """ returns tuple (input, output) """

    i = idx * self.batch_size
    batch_input_imgs = self.input_imgs[i : i + self.batch_size]
    batch_output_imgs = self.output_imgs[i : i + self.batch_size]

    print(self.img_size)
    x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
    for z, path in enumerate(batch_input_imgs):
      img = load_img(path, target_size=self.img_size)
      x[z] = img

    y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
    for z, path in enumerate(batch_output_imgs):
      img = load_img(path, target_size=self.img_size)
      y[z] = img

    return x, y
