from tensorflow.keras import layers

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
num_classes = 8

kernel_size = (3, 3)

# rgb inputs
input = layers.Input(shape=(img_height, img_width, 3))

x = layers.Conv2D(32, 3, strides=2, padding='same')(input)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
