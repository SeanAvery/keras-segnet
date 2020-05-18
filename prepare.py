import os

data_file = 'comma10k/files_trainable'
input_dir = 'comma10k/imgs'
output_dir = 'comma10k/masks'
img_size = (1164, 874)
num_classes = 6

with open(data_file, 'r') as data: 
  files = data.read().splitlines()

input_imgs = map(lambda path: 'comma10k/{}'.format(path), files)

output_imgs = map(lambda path: 'comma10k/imgs/{}'.format(path.split('/')[1]), files)
