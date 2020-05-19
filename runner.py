from prepare import prepare
from model import build_model
from data import Dataset

# prepare the data set
print('preparing dataset')
train_inputs, train_outputs, test_inputs, test_outputs = prepare()
train_data = Dataset(32, (1164, 874), test_inputs, test_outputs)
test_data = Dataset(32, (1164, 874), train_inputs, train_outputs)

# build model
print('building model')
model = build_model()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# run training and testing
epochs = 10
model.fit(train_data, epochs=epochs, validation_data=test_data)
