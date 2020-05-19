def prepare():
  data_file = 'comma10k/files_trainable'
  input_dir = 'comma10k/imgs'
  output_dir = 'comma10k/masks'
  img_size = (1164, 874)
  num_classes = 6

  with open(data_file, 'r') as data: 
    files = data.read().splitlines()

  input_imgs = list(map(lambda path: 'comma10k/{}'.format(path), files))
  output_imgs = list(map(lambda path: 'comma10k/imgs/{}'.format(path.split('/')[1]), files))
  
  test_num = 300
  train_inputs = input_imgs[:-test_num]
  train_outputs = output_imgs[:-test_num]
  test_inputs = input_imgs[-test_num:]
  test_outputs = output_imgs[-test_num:]

  return (train_inputs, train_outputs, test_inputs, test_outputs)
