from prepare import prepare
from model import build_model
from data import Dataset

# prepare the data set
print('preparing dataset')
inputs, outputs = prepare()
data = Dataset(32, (1164, 874), inputs, outputs)

# build model
print('building model')
model = build_model()
mdoel.compile(optmizer='adam', loss='sparse_categorical_crossentropy')

# run training

# run testing
