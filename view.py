import cv2
import time
import numpy as np

root_path = 'comma10k'

with open('{}/files_trainable'.format(root_path), 'r') as f:
  files = f.read().splitlines()

for file in files:
  img_id = file.split('/')[1]
  print(img_id)
  raw = cv2.imread('{0}/imgs/{1}'.format(root_path, img_id))
  mask = cv2.imread('{0}/masks/{1}'.format(root_path, img_id))
  print(mask)
  cv2.imshow('mask', mask)
  cv2.imshow('raw', raw)
  cv2.waitKey(1000)
  cv2.destroyAllWindows()
