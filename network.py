from keras.models import Sequential
from keras.layers import Dense#, Embedding
# from keras.layers import LSTM
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow import keras
# import keras
# from keras_preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
from keras.layers import  Flatten, Dropout, Activation #Input,
from keras.layers import Conv1D, MaxPooling1D#, AveragePooling1D
# from keras.models import Model
# from keras.callbacks import ModelCheckpoint
# from sklearn.metrics import confusion_matrix
from keras import optimizers

def SER_CNN(input_shape):
    model = Sequential()

    model.add(Conv1D(128, 5,padding='same',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5,padding='same',))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
  

    return model

