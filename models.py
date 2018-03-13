from keras.layers import Dense, Flatten, Dropout, Conv1D, Reshape, LeakyReLU, MaxPool1D
from keras.models import Sequential


def createModelWithEncoder(encoder, outputShape):
    inputShape = encoder.layers[-1].output_shape[1:]

    model = Sequential()
    for layer in encoder.layers:
        model.add(layer)

    model.add(Dropout(.2, input_shape=inputShape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(outputShape))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def createModel(inputShape, outputShape):
    model = Sequential()
    model.add(Dropout(.2, input_shape=inputShape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(outputShape))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def createConvModel(inputShape, outputShape):
    model = Sequential()
    model.add(Dropout(.2, input_shape=inputShape))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(LeakyReLU())
    model.add(MaxPool1D())
    model.add(Dropout(.5))
    model.add(Conv1D(1024, 3, padding='same', activation='relu'))
    model.add(LeakyReLU())
    model.add(MaxPool1D())
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(outputShape))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Larger model, takes longer to stabilize
# def createConvModel(inputShape, outputShape):
#     model = Sequential()
#     model.add(Dropout(.2, input_shape=inputShape))
#     model.add(Conv1D(1024, 3, padding='same', activation='relu'))
#     model.add(LeakyReLU())
#     model.add(MaxPool1D())
#     model.add(Dropout(.2))
#     model.add(Conv1D(128, 3, padding='same', activation='relu'))
#     model.add(LeakyReLU())
#     model.add(MaxPool1D())
#     model.add(Dropout(.5))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(.2))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(8, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(outputShape))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model
