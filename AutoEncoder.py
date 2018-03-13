from keras.layers import Input, Dropout, Conv2D, MaxPool2D, UpSampling2D, Reshape
from keras.models import Model


class AutoEncoder():
    def __init__(self, inputShape):
        kernelSize = (3, inputShape[1])

        # Create autoencoder model
        input = Input(inputShape)

        shapeWithOneAdded = inputShape + tuple([1])
        reshaped = Reshape(shapeWithOneAdded)(input)

        dropout = Dropout(.2, input_shape=inputShape)(reshaped)

        conv0 = self.convBlock(16, dropout, kernelSize, maxpool=True)
        conv1 = self.convBlock(8, conv0, kernelSize)

        deconv0 = self.deconvBlock(8, conv1, kernelSize)
        deconv1 = self.deconvBlock(16, deconv0, kernelSize, unpool=True)
        deconv2 = self.deconvBlock(1, deconv1, kernelSize)

        # Remove "color" dimension required for Conv2D
        output = Reshape(inputShape)(deconv2)

        model = Model(input, output)
        model.compile(loss='mean_squared_error', optimizer='adam')

        encoderInput = Input(inputShape)
        currentTensor = encoderInput
        for i in range(1, 6):
            nextLayer = model.layers[i]
            if type(nextLayer) is not Dropout:
                currentTensor = nextLayer(currentTensor)

        encoder = Model(encoderInput, currentTensor, name='Encoder')
        encoder.compile(loss='mean_squared_error', optimizer='adam')

        self.model = model
        self.encoder = encoder

    def convBlock(self, filters, input, kernelSize, maxpool=False):
        conv = Conv2D(filters, kernelSize, padding='same', activation='relu')(input)
        if maxpool:
            pool = MaxPool2D((2, 1))(conv)
            return pool
        else:
            return conv

    def deconvBlock(self, filters, input, kernelSize, unpool=False):
        if unpool:
            input = UpSampling2D((2, 1))(input)
        deconv = Conv2D(filters, kernelSize, padding='same')(input)
        return deconv

    def fit(self, trainData, testData):
        self.model.fit(trainData, trainData, epochs=300, batch_size=256, validation_data=[testData, testData])
        return self.encoder
