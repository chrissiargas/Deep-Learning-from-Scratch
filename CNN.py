import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import sklearn.datasets
import copy
import pickle
from scipy.interpolate import make_interp_spline
from buildNeuralNetwork import NN
import gzip

tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.datasets import load_sample_image

def to_tensor(npArray):
    tensor = tf.convert_to_tensor(npArray,dtype=tf.float32)
    return tensor




class Conv2D:
    def __init__(self,filters=1,kernel_size=1,channels=1,strides=1,padding=False,pads=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.channels = channels
        self.strides = strides
        self.padding = padding
        self.pads = pads

class maxPooling:
    def __init__(self,poolingSize,stride=1):
        self.poolingSize = poolingSize
        self.stride = stride

class Flatten:
    def __init__(self):
        pass

class NeuralNetwork:
    def __init__(self,
                 layers=None,
                 activations=None,
                 initializations=None,
                 gradientClipping=False,
                 gradientClippingThreshold=1.0,
                 batch_size=None,
                 loss_function="MSE",
                 manipulate=None,
                 batch_normalization=False,
                 batch_norm_momentum=0.9,
                 optimizer=None,
                 momentum=0.9,
                 decay_rate=0.9,
                 beta1=0.9,
                 beta2=0.999,
                 l1Regularization=False,
                 l2Regularization=False,
                 Lambda=0.01,
                 dropout=False,
                 dropoutProb=0.5,
                 shuffle=False,
                 delta=1.0):

        self.layers = layers
        self.activations = activations
        self.loss_function = loss_function
        self.batch_normalization = batch_normalization
        self.optimizer = optimizer


class CNN:

    def __init__(self,images,labels,layers):
        self.images = to_tensor(images)
        self.labels = to_tensor(labels)

        self.convLayers = []
        self.biasLayers = []

        self.layers = layers

        self.epoch = 0


        for index,layer in enumerate(layers):
            if isinstance(layer,Conv2D):
                newConvLayer = tf.random.normal(shape=[layer.filters,layer.channels,layer.kernel_size,layer.kernel_size])
                newBiasLayer = tf.zeros(shape=[layer.filters])
                self.convLayers.append(newConvLayer)
                self.biasLayers.append(newBiasLayer)

            if isinstance(layer,NeuralNetwork):
                test = CNN([images[0]],[labels[0]],layers[:index])
                predOutput = test.feedForward([images[0]])
                inputSize = predOutput.shape[1]
                outputSize = labels.shape[1]
                print(predOutput)
                print(inputSize)
                print(outputSize)



                self.outputNN = NN(layers=layer.layers,
                                   activations=layer.activations,
                                   loss_function=layer.loss_function,
                                   batch_normalization=layer.batch_normalization,
                                   optimizer=layer.optimizer,
                                   inputSize=inputSize,
                                   outputSize=outputSize)

                self.optimizer = self.layers[-1].optimizer
                self.parameters = [[self.convLayers, self.biasLayers], self.outputNN.parameters]





    def padding(self,inputs,pad):
        return tf.pad(inputs, [[0,0],[0,0],pad[0:2],pad[2:4]], "CONSTANT")

    def convolution(self,inputs,filters,biases,stride=1,padding=False,pads=None):


        (n_f, n_c_f, f, _) = filters.shape  # filter dimensions

        if padding :
            if not pads:
                padSize = f - 1
                if f%2!=0:
                    inputs = self.padding(inputs,[padSize//2 for _ in range(4)])

                else:
                    inputs = self.padding(inputs,
                        [tf.math.ceil(padSize/2),tf.math.ceil(padSize/2),tf.math.floor(padSize/2),tf.math.floor(padSize/2)])

            else:
                inputs = self.padding(inputs,pads)


        n, n_c, in_dim, _ = inputs.shape  # image dimensions

        out_dim = int((in_dim - f) / stride) + 1  # calculate output dimensions
        out = tf.zeros([n,n_f,out_dim,out_dim],tf.float32)

        for input_index,input in enumerate(inputs):
            for f_index,filter,bias in enumerate(zip(filters,biases)):
               for out_y,curr_y in enumerate(range(0,in_dim-f+1,stride)):
                   for out_x,curr_x in enumerate(range(0,in_dim-f+1,stride)):
                       out[input_index,f_index, out_y, out_x] = tf.reduce_sum(filter * input[:,curr_y:curr_y+f,curr_x:curr_x+f])+bias

        return out

    def maxPool(self,input,poolingSize=2,stride=2):
        f = poolingSize

        n_c, h_input, w_input = input.shape

        h,w = int((h_input-f)/stride) + 1, int((w_input-f)/stride)

        out = tf.zeros([n_c,h,w],tf.float32)

        for channelIndex in range(n_c):
            for out_y,curr_y in enumerate(range(0,h-f+1,stride)):
                for out_x,curr_x in enumerate(range(0,w-f+1,stride)):
                    out[channelIndex, out_y, out_x] = tf.reduce_max(input[channelIndex, curr_y:curr_y+f, curr_x:curr_x+f])

        return out

    def flattenInput(self,inputs):
        n, n_c, dim, _ = inputs.shape
        out = input.reshape([n, n_c*dim*dim])

    def feedForward(self,input):


        convIndex = 0


        for layer in self.layers:
            if isinstance(layer,Conv2D):

                input=self.convolution(inputs=input,filters=self.convLayers[convIndex],biases=self.biasLayers[convIndex],stride=layer.strides,padding=layer.padding,pads=layer.pads)
                convIndex+=1

            elif isinstance(layer,maxPooling):

                input = self.maxPool(input=input,poolingSize=layer.poolingSize,stride=layer.stride)

            elif isinstance(layer,Flatten):

                input = self.flattenInput(input)

            elif isinstance(layer,NeuralNetwork):
                input = self.outputNN.feedForward(input,training=True)

        output = input
        return output

    def loss(self,labels,predLabels):
        return self.outputNN.loss(labels,predLabels)


    def findVelocity(self,dError_dWeights,lr, index):

        if not self.epoch == 0:
            velocity = self.velocity[index]



        if not self.optimizer:

            if self.epoch == 0:

                velocity = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    velocity.append([])
                    for elP in dParameter:
                        velocity[dIndex].append(-lr * elP)


            else:
                for dIndex, dParameter in enumerate(dError_dWeights):
                    for elIndex, elP in enumerate(dParameter):
                        velocity[dIndex][elIndex] = - lr * elP


        elif self.optimizer == "momentum" or self.optimizer == "nesterov":

            if self.epoch == 0:

                velocity = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    velocity.append([])
                    for el in dParameter:
                        velocity[dIndex].append(-lr * el)



            else:

                for dIndex, (dParameter, dVelocity) in enumerate(zip(dError_dWeights, velocity)):
                    for elIndex, (elP, elV) in enumerate(zip(dParameter, dVelocity)):
                        velocity[dIndex][elIndex] = self.outputNN.momentum * elV - lr * elP



        elif self.optimizer == "adagrad":
            zero = 1e-8

            if self.epoch == 0:

                velocity = []
                sumGrad = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    velocity.append([])
                    sumGrad.append([])
                    for elIndex, el in enumerate(dParameter):
                        sumGrad[dIndex].append(el * el)
                        velocity[dIndex].append(-lr * el / tf.sqrt(sumGrad[dIndex][elIndex] + zero))


            else:
                sumGrad = self.sumGrad[index]

                for dIndex, (dParameter, dVelocity, dSG) in enumerate(
                        zip(dError_dWeights, velocity, sumGrad)):
                    for elIndex, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
                        sumGrad[dIndex][elIndex] += elP * elP
                        velocity[dIndex][elIndex] = -lr * elP / tf.sqrt(elSG + zero)


        elif self.optimizer == "RMSProp":

            zero = 1e-8

            if self.epoch == 0:

                velocity = []
                sumGrad = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    velocity.append([])
                    sumGrad.append([])
                    for elIndex, el in enumerate(dParameter):
                        sumGrad[dIndex].append((1 - self.outputNN.decayRate) * el * el)
                        velocity[dIndex].append(-lr * el / tf.sqrt(sumGrad[dIndex][elIndex] + zero))



            else:
                sumGrad = self.sumGrad[index]

                for dIndex, (dParameter, dVelocity, dSG) in enumerate(
                        zip(dError_dWeights, velocity, sumGrad)):
                    for elIndex, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
                        sumGrad[dIndex][elIndex] = self.outputNN.decayRate * elSG + (
                                    1 - self.outputNN.decayRate) * elP * elP
                        velocity[dIndex][elIndex] = -lr * elP / tf.sqrt(elSG + zero)

        elif self.optimizer == "adam" or self.optimizer == "nadam":

            zero = 1e-8

            if self.epoch == 0:

                velocity = []
                sumMom = []
                sumMomHat = []
                sumGrad = []
                sumGradHat = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    velocity.append([])
                    sumGrad.append([])
                    sumGradHat.append([])
                    sumMom.append([])
                    sumMomHat.append([])

                    for el in dParameter:
                        sumMom[dIndex].append(-(1 - self.outputNN.beta1) * el)
                        sumMomHat[dIndex].append(el)
                        sumGrad[dIndex].append((1 - self.outputNN.beta2) * el * el)
                        sumGradHat[dIndex].append(el * el)
                        velocity[dIndex].append(lr * el / tf.sqrt(el * el + zero))

                    dIndex += 1

            else:
                velocity = self.velocity[index]
                sumMom = self.sumMom[index]
                sumMomHat = self.sumMomHat[index]
                sumGrad = self.sumGrad[index]
                sumGradHat = self.sumGradHat[index]

                for dIndex, (dParameter, dVelocity, dSG, dSM) in enumerate(
                        zip(dError_dWeights, velocity, sumGrad, sumMom)):
                    for index, (elP, elV, elSG, elSM) in enumerate(zip(dParameter, dVelocity, dSG, dSM)):
                        sumMom[dIndex][index] = self.outputNN.beta1 * elSM - (1 - self.outputNN.beta1) * elP
                        sumGrad[dIndex][index] = self.outputNN.beta2 * elSG + (1 - self.outputNN.beta2) * elP * elP
                        sumMomHat[dIndex][index] = sumMom[dIndex][index] / (
                                1 - self.outputNN.beta1 ** (self.epoch + 1))
                        sumGradHat[dIndex][index] = sumGrad[dIndex][index] / (
                                1 - self.outputNN.beta1 ** (self.epoch + 1))
                        velocity[dIndex][index] = lr * sumMomHat[dIndex][index] / tf.sqrt(
                            sumGradHat[dIndex][index] + zero)



        else:

            if self.epoch == 0:

                velocity = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    velocity.append([])
                    for elP in dParameter:
                        velocity[dIndex].append(-lr * elP)


            else:
                for dIndex, dParameter in enumerate(dError_dWeights):
                    for elIndex, elP in enumerate(dParameter):
                        velocity[dIndex][elIndex] = - lr * elP

        if self.outputNN.gradientClipping:
            norm = tf.norm(velocity)
            if norm > self.outputNN.GCT:
                velocity *= self.outputNN.GCT / norm

        if self.epoch == 0:
            self.velocity.append(velocity)

            try:
                self.sumGrad.append(sumGrad)
            except:
                pass

            try:
                self.sumGradHat.append(sumGradHat)
            except:
                pass

            try:
                self.sumMom.append(sumMom)
            except:
                pass

            try:
                self.sumMomHat.append(sumMomHat)
            except:
                pass

        else:
            self.velocity[index] = velocity

            try:
                self.sumGrad[index] = sumGrad
            except:
                pass

            try:
                self.sumGradHat[index] = sumGradHat
            except:
                pass

            try:
                self.sumMom[index] = sumMom
            except:
                pass

            try:
                self.sumMomHat[index] = sumMomHat
            except:
                pass



    def backPropagation(self,t,error,lr):
         dError_dWeights_all = t.gradient(error,self.parameters)

         self.findVelocity(dError_dWeights_all[0],lr,0)
         self.findVelocity(dError_dWeights_all[1],lr,1)


         #
         # if not self.optimizer:
         #
         #     if self.epoch == 0:
         #
         #         self.velocity = []
         #
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #
         #             self.velocity.append([])
         #             for elP in dParameter:
         #                 self.velocity[dIndex].append(-lr * elP)
         #
         #
         #     else:
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #             for elIndex, elP in enumerate(dParameter):
         #                 self.velocity[dIndex][elIndex] = - lr * elP
         #
         #
         # elif self.optimizer == "momentum" or self.optimizer == "nesterov":
         #
         #     if self.epoch == 0:
         #
         #         self.velocity = []
         #
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #
         #             self.velocity.append([])
         #             for el in dParameter:
         #                 self.velocity[dIndex].append(-lr * el)
         #
         #
         #
         #     else:
         #
         #         for dIndex, (dParameter, dVelocity) in enumerate(zip(dError_dWeights, self.velocity)):
         #             for elIndex, (elP, elV) in enumerate(zip(dParameter, dVelocity)):
         #                 self.velocity[dIndex][elIndex] = self.outputNN.momentum * elV - lr * elP
         #
         #
         #
         # elif self.optimizer == "adagrad":
         #     zero = 1e-8
         #
         #     if self.epoch == 0:
         #
         #         self.velocity = []
         #         self.sumGrad = []
         #
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #
         #             self.velocity.append([])
         #             self.sumGrad.append([])
         #             for elIndex, el in enumerate(dParameter):
         #                 self.sumGrad[dIndex].append(el * el)
         #                 self.velocity[dIndex].append(-lr * el / tf.sqrt(self.sumGrad[dIndex][elIndex] + zero))
         #
         #
         #     else:
         #
         #         for dIndex, (dParameter, dVelocity, dSG) in enumerate(
         #                 zip(dError_dWeights, self.velocity, self.sumGrad)):
         #             for elIndex, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
         #                 self.sumGrad[dIndex][elIndex] += elP * elP
         #                 self.velocity[dIndex][elIndex] = -lr * elP / tf.sqrt(elSG + zero)
         #
         #
         # elif self.optimizer == "RMSProp":
         #
         #     zero = 1e-8
         #
         #     if self.epoch == 0:
         #
         #         self.velocity = []
         #         self.sumGrad = []
         #
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #
         #             self.velocity.append([])
         #             self.sumGrad.append([])
         #             for elIndex, el in enumerate(dParameter):
         #                 self.sumGrad[dIndex].append((1 - self.decayRate) * el * el)
         #                 self.velocity[dIndex].append(-lr * el / tf.sqrt(self.sumGrad[dIndex][elIndex] + zero))
         #
         #
         #
         #     else:
         #
         #         for dIndex, (dParameter, dVelocity, dSG) in enumerate(
         #                 zip(dError_dWeights, self.velocity, self.sumGrad)):
         #             for elIndex, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
         #                 self.sumGrad[dIndex][elIndex] = self.outputNN.decayRate * elSG + (1 - self.outputNN.decayRate) * elP * elP
         #                 self.velocity[dIndex][elIndex] = -lr * elP / tf.sqrt(elSG + zero)
         #
         # elif self.optimizer == "adam" or self.optimizer == "nadam":
         #
         #     zero = 1e-8
         #
         #     if self.epoch == 0:
         #
         #         self.velocity = []
         #         self.sumMom = []
         #         self.sumMomHat = []
         #         self.sumGrad = []
         #         self.sumGradHat = []
         #
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #
         #             self.velocity.append([])
         #             self.sumGrad.append([])
         #             self.sumGradHat.append([])
         #             self.sumMom.append([])
         #             self.sumMomHat.append([])
         #
         #             for el in dParameter:
         #                 self.sumMom[dIndex].append(-(1 - self.outputNN.beta1) * el)
         #                 self.sumMomHat[dIndex].append(el)
         #                 self.sumGrad[dIndex].append((1 - self.outputNN.beta2) * el * el)
         #                 self.sumGradHat[dIndex].append(el * el)
         #                 self.velocity[dIndex].append(lr * el / tf.sqrt(el * el + zero))
         #
         #             dIndex += 1
         #
         #     else:
         #         for dIndex, (dParameter, dVelocity, dSG, dSM) in enumerate(
         #                 zip(dError_dWeights, self.velocity, self.sumGrad, self.sumMom)):
         #             for index, (elP, elV, elSG, elSM) in enumerate(zip(dParameter, dVelocity, dSG, dSM)):
         #                 self.sumMom[dIndex][index] = self.beta1 * elSM - (1 - self.outputNN.beta1) * elP
         #                 self.sumGrad[dIndex][index] = self.beta2 * elSG + (1 - self.outputNN.beta2) * elP * elP
         #                 self.sumMomHat[dIndex][index] = self.sumMom[dIndex][index] / (
         #                         1 - self.beta1 ** (self.epoch + 1))
         #                 self.sumGradHat[dIndex][index] = self.sumGrad[dIndex][index] / (
         #                         1 - self.beta1 ** (self.epoch + 1))
         #                 self.velocity[dIndex][index] = lr * self.sumMomHat[dIndex][index] / tf.sqrt(
         #                     self.sumGradHat[dIndex][index] + zero)
         #
         #
         #
         # else:
         #
         #     if self.epoch == 0:
         #
         #         self.velocity = []
         #
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #
         #             self.velocity.append([])
         #             for elP in dParameter:
         #                 self.velocity[dIndex].append(-lr * elP)
         #
         #
         #     else:
         #         for dIndex, dParameter in enumerate(dError_dWeights):
         #             for elIndex, elP in enumerate(dParameter):
         #                 self.velocity[dIndex][elIndex] = - lr * elP
         #
         # if self.outputNN.gradientClipping:
         #     norm = tf.norm(self.velocity)
         #     if norm > self.outputNN.GCT:
         #         self.velocity *= self.outputNN.GCT / norm

    def update_parameters(self):
        for index in range(self.convLayers):
            self.convLayers[index] += self.velocity[0][0][index]

        for index in range(self.biasLayers):
            self.biasLayers[index] += self.velocity[0][1][index]

        self.outputNN.update_parameters(self.velocity[1])

        self.parameters = [[self.convLayers,self.biasLayers],self.outputNN.parameters]

    def moveOneStep(self):

        for Index, (P,V) in enumerate(zip(self.parameters,self.velocity)):
            for dIndex, (dParameter, dVelocity) in enumerate(zip(P, V)):
                for elIndex, (elP, elV) in enumerate(zip(dParameter, dVelocity)):
                    self.parameters[Index][dIndex][elIndex] = (elP + self.outputNN.momentum * elV)



    def train(self,lr=0.01,epochs=400):

        for epoch in epochs:

            with tf.GradientTape() as t:

                if self.optimizer == "nesterov" or self.optimizer == "nadam":
                    if self.epoch != 0:
                        self.moveOneStep()

                t.watch(self.parameters)


                predLabels = self.feedForward(self.images)
                error = self.loss(self.labels, predLabels)

                print(error)

                self.backPropagation(t,error,lr)
                self.update_parameters()

            self.epoch += 1

    def __str__(self):

        string = ""
        for index,convLayer,biasLayer in enumerate(zip(self.convLayers,self.biasLayers)):
            string += "Kernels:" + str(index + 1) + "\n"
            string += "Convolution Layer" + "\n"
            for filter in convLayer:
                for channel in filter:
                    for y in channel:
                        string += "["
                        for i,x in enumerate(y):
                            if i != 0:
                                string += ","
                            string += str(x.numpy())
                        string += "]" + "\n"
                    string += "\n"
                string += "\n"
            string += "\n"

            string += "Biases Layer" + "\n"
            string += "["
            for i,bias in enumerate(biasLayer):
                if i != 0:
                    string += ","
                string += str(bias.numpy())
            string += "]" + "\n\n"

        string += "Final Neural Network:" + "\n"
        string += self.outputNN.__str__()

        return string

def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m
    is the number of training examples.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

from matplotlib import pyplot as plt


m =50000

X = extract_data('train-images-idx3-ubyte.gz', m, 28)
plt.imshow(X[0].reshape([28,28]), interpolation='nearest')
plt.show()

y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
print(y_dash[0])

X-= int(np.mean(X))
X/= int(np.std(X))
plt.imshow(X[0].reshape([28,28]), interpolation='nearest')
plt.show()



layers = []
newLayer = Conv2D(filters=4,kernel_size=3,channels=1,padding=True)
layers.append(newLayer)
newLayer = maxPooling(2,stride=2)
layers.append(newLayer)
newLayer = Conv2D(filters=4,kernel_size=3,channels=1,padding=True)
layers.append(newLayer)
newLayer = maxPooling(2,stride=2)
layers.append(newLayer)
newLayer = Conv2D(filters=4,kernel_size=3,channels=1,padding=True)
layers.append(newLayer)
newLayer = maxPooling(2,stride=2)
layers.append(newLayer)

Layers = [10,10,10,10]
newLayer = NeuralNetwork(
       Layers,
       loss_function="CCE",
       activations=["selu","selu","selu","selu","softmax"],
       initializations="lecun_normal",
       batch_normalization=True,
       optimizer="nadam"
       )
layers.append(newLayer)


A = CNN(images=X[:300],
        labels=y_dash[:300],
        layers=layers)


