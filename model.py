from tensorflow import keras
from tensorflow.keras import layers

def build_model():
  img_height = 1164
  img_width = 874

  """
   0 - #000000 - empty
   1 -         - sky (deprecated, now undrivable)
   2 - #402020 - road (all parts, anywhere nobody would look at you funny for driving)
   3 - #ff0000 - lane markings (don't include non lane markings like turn arrows and crosswalks)
   4 - #808060 - undrivable
   5 - #00ff66 - movable (split into vehicles and people/animals?, actually don't)
   6 -         - signs and traffic lights (deprecated, now undrivable)
   7 - #cc00ff - my car (and anything inside it, including wires, mounts, etc...)
  """
  num_classes = 6

  kernel_size = (3, 3)

  # rgb inputs
  input = layers.Input(shape=(img_height, img_width, 3))

  x = layers.Conv2D(32, 3, strides=2, padding='same')(input)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  previous_block = x

  """
    encoder:
    downsample input rgb image
  """
  for filter in [64, 128, 256]:
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(filters=filter, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(filters=filter, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Project residual
    residual = layers.Conv2D(filter, 1, strides=2, padding="same")(
        previous_block
    )
    x = layers.add([x, residual])  # Add back residual
    previous_block = x  # Set aside next residual

  """
    decoder:
    upsample back to full scale image size
  """ 
  previous_block = x  # Set aside residual
  for filter in [256, 128, 64, 32]:
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=filter, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters=filter, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.UpSampling2D(2)(x)
    
    residual = layers.UpSampling2D(2)(previous_block)
    residual = layers.Conv2D(filter, 1, padding="same")(residual)
    x = layers.add([x, residual])  # Add back residual
    previous_block = x  # Set aside next residual

  # output segmented image
  output = layers.Conv2D(num_classes, kernel_size=kernel_size, activation='softmax', padding='same')(x)

  # define da model
  model = keras.Model(input, output)

  # clear ram
  keras.backend.clear_session()

  # build da model
  model.summary()
  
  return model 
