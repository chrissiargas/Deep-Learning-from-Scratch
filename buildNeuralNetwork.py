import matplotlib.pyplot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import sklearn.datasets
import copy
import pickle
from scipy.interpolate import make_interp_spline
from matplotlib import cm


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "svm"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

tf.config.optimizer.set_jit(True)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
global listOfColors
listOfColors = ['b','g','r','c','m','y']

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def to_tensor(npArray):
    tensor = tf.convert_to_tensor(npArray,dtype=tf.float32)
    return tensor


def pow2(x):
    return x**2

# x1 = np.full((2,2),1)
# x2 = np.full((2,2),2)
#
# x1 = to_tensor(x1)
# x2 = to_tensor(x2)
#
# x = tf.stack([x1,x2])
# print(x[0])
#
# for _ in range(10000):
#     with tf.GradientTape() as t:
#         t.watch(x)
#
#
#         k = tf.reduce_sum(x)
#
#         y = 2*k
#         z = tf.multiply(y,y)
#
#
#
#     dz_dx = t.gradient(z,x)
#     x -= 0.01*dz_dx*x
#
# print(x)

tf.random.set_seed = 42


class NN:
    def __init__(self,
                 data,
                 targets,
                 layers,
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

        self.produceData = manipulate
        self.numberOfLayers = len(layers) + 1


        if len(data.shape)==1:
            data = np.reshape(data,(-1,1))



        if self.produceData:
            self.x = to_tensor(self.manipulate_data(data,columns=manipulate[0],func=manipulate[1]))

        else:
            self.x = to_tensor(data)





        if len(targets.shape)==1:
            targets = np.reshape(targets,(-1,1))

        self.y = to_tensor(targets)

        if shuffle:
            indices = tf.range(start=0,limit=tf.shape(self.x)[0],dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)


            self.x = tf.gather(self.x, shuffled_indices)
            self.y = tf.gather(self.y, shuffled_indices)



        if batch_size:
            assert isinstance(batch_size, int)

            if batch_size>len(self.x) or batch_size<=0:
                self.batch_size = len(self.x)

            else:
                self.batch_size = batch_size

        else:
            self.batch_size = len(self.x)


        if not activations:
            self.activations = ['linear' for _ in range(self.numberOfLayers)]

        elif isinstance(activations, str):
            self.activations = [activations for _ in range(self.numberOfLayers)]

        else:
            assert len(activations) == self.numberOfLayers and all(isinstance(el, str) for el in activations)

            self.activations = activations


        if not initializations:
            self.initializations = ['xavier_uniform' for _ in range(self.numberOfLayers)]

        elif isinstance(initializations, str):
            self.initializations = [initializations for _ in range(self.numberOfLayers)]

        else:
            assert len(initializations) == self.numberOfLayers and all(isinstance(el, str) for el in initializations)
            self.initializations = initializations


        #exploding/vanishing gradient
        self.gradientClipping = gradientClipping
        self.GCT = gradientClippingThreshold


        self.lossFunction = loss_function
        self.delta = delta


        if isinstance(batch_normalization,bool):
            self.batchNormalization = [batch_normalization for _ in range(self.numberOfLayers)]

        else:
            assert len(batch_normalization) == self.numberOfLayers and all(isinstance(el,bool) for el in batch_normalization)
            self.batchNormalization = batch_normalization

        self.batch_norm_momentum = batch_norm_momentum


        self.optimizer = optimizer
        self.momentum = momentum
        self.decayRate = decay_rate
        self.beta1 = beta1
        self.beta2 = beta2


        self.epoch = 0


        self.layerNeurons = copy.deepcopy(layers)

        self.layerNeurons.insert(0,self.x.shape[1])


        self.Layers = []
        self.biases = []



        self.BatchLayersGamma = []
        self.BatchLayersBeta = []

        if type(l1Regularization) == list:
            assert len(l1Regularization) == self.numberOfLayers and all(isinstance(el,bool) for el in l1Regularization)
            self.l1Regularization = l1Regularization

        else:
            self.l1Regularization = []
            for _ in range(self.numberOfLayers):
                self.l1Regularization.append(l1Regularization)

        if type(l2Regularization) == list:
            assert len(l2Regularization) == self.numberOfLayers and all(isinstance(el,bool) for el in l2Regularization)
            self.l2Regularization = l2Regularization

        else:
            self.l2Regularization = []
            for _ in range(self.numberOfLayers):
                self.l2Regularization.append(l2Regularization)

        assert isinstance(Lambda,float)
        self.Lambda = Lambda

        if type(dropout) == list:
            assert len(dropout) == self.numberOfLayers and all(isinstance(el,bool) for el in dropout)
            self.dropout = dropout

        else:
            self.dropout = []
            for _ in range(self.numberOfLayers):
                self.dropout.append(dropout)

        if type(dropoutProb) == list:
              assert len(dropoutProb) == self.numberOfLayers and all(isinstance(el, float) for el in dropoutProb)
              self.dropoutProb = dropoutProb

        else:
            self.dropoutProb = []
            for _ in range(self.numberOfLayers):
                self.dropoutProb.append(dropoutProb)


        nextLayer = targets.shape[1]

        for layer,initializer,batchNormLayer in zip(self.layerNeurons[::-1],self.initializations[::-1],self.batchNormalization[::-1]):
            rng = tf.random.experimental.Generator.from_seed(41)
            # exploding/vanishing gradient
            if initializer=="xavier_uniform": #tanh,sigmoid,logistic
                newLayer = rng.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)
                newLayer *= tf.sqrt(6/(layer+nextLayer))

            elif initializer=="xavier_normal":
                newLayer = rng.normal(shape=[layer, nextLayer], dtype=tf.float32)
                newLayer *= tf.sqrt(2/(layer+nextLayer))

            elif initializer=="he_uniform": #relu and variables
                newLayer = rng.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)
                newLayer *= tf.sqrt(6/layer)

            elif initializer=="he_normal":
                newLayer = rng.normal(shape=[layer, nextLayer], dtype=tf.float32)
                newLayer *= tf.sqrt(2 / layer)

            elif initializer=="lecun_uniform": #selu
                newLayer = rng.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)
                newLayer *= tf.sqrt(3 / layer)

            elif initializer=="lecun_normal":
                newLayer = rng.normal(shape=[layer, nextLayer], dtype=tf.float32)
                newLayer *= tf.sqrt(1 / layer)

            else:
                newLayer = rng.normal(shape=[layer, nextLayer], dtype=tf.float32)

            self.Layers.append(newLayer)

            self.biases.append(tf.zeros(shape=[nextLayer],dtype=tf.dtypes.float32))

            if batchNormLayer:
                newBatchNormLayerGamma = tf.ones(shape=[nextLayer],dtype=tf.dtypes.float32)
                newBatchNormLayerBeta = tf.zeros(shape=[nextLayer],dtype=tf.dtypes.float32)


                self.BatchLayersGamma.append(newBatchNormLayerGamma)
                self.BatchLayersBeta.append(newBatchNormLayerBeta)

            nextLayer = layer

        self.Layers.reverse()
        self.biases.reverse()

        self.BatchLayersGamma.reverse()
        self.BatchLayersBeta.reverse()

        self.parameters = [self.Layers,self.BatchLayersGamma,self.BatchLayersBeta,self.biases]

    # def __setattr__(self, key, value):
    #     if key == 'parameters':
    #         self.__dict__[key] = copy.deepcopy(value)
    #         self.Layers = self.parameters[0]
    #         self.BatchLayersGamma = self.parameters[1]
    #         self.BatchLayersBeta = self.parameters[2]
    #         self.biases = self.parameters[3]
    #
    #     if key == 'x':
    #         raise AttributeError('x is private')
    #
    #     if key == 'y':
    #         raise  AttributeError('y is private')
    #
    #     if key == 'Layers':
    #         raise AttributeError('Layers are private')
    #



    def manipulate_data(x, columns=None, func=None):
        data = []

        if columns == None or func == None:
            return

        for index in range(len(x)):

            data.append(np.append(x[index], func(x[index][columns])))

        data = np.array(data)
        return data


    def __str__(self):
        string = ""
        batchIndex = 0
        for index,(weights,biases,BatchLayer) in enumerate(zip(self.Layers,self.biases,self.batchNormalization)):

            string+="Weights:"+str(index+1)+"\n"
            for weight in weights:
                string+="["
                for i,element in enumerate(weight):
                    if i!=0:
                        string+=","
                    string+=str(element.numpy())
                string+="]"+"\n"
            string+="\n"

            string += "Biases:" + str(index + 1) + "\n"
            string += "["
            for i, bias in enumerate(biases):

                if i != 0:
                    string += ","
                string += str(bias.numpy())

            string += "]"
            string += "\n\n"

            if BatchLayer:
                string += "Gammas:" + str(index + 1) + "\n"
                string+="["
                for i,gamma in enumerate(self.BatchLayersGamma[batchIndex]):

                    if i!=0:
                        string+=","
                    string+=str(gamma.numpy())

                string+="]"
                string+="\n\n"

                string += "Betas:" + str(index + 1) + "\n"
                string+="["
                for i, beta in enumerate(self.BatchLayersBeta[batchIndex]):

                    if i != 0:
                        string += ","
                    string += str(beta.numpy())

                string+="]"
                batchIndex += 1
            string+="\n\n\n"



        return string

    def batch_norm(self, x, gamma, beta, scope_name, is_training=True, debug=False):
        eps = tf.constant(1e-5)
        momentum = tf.constant(self.batch_norm_momentum)

        try:
            self._BN_MOVING_VARS, self._BN_MOVING_MEANS

        except:

            self._BN_MOVING_VARS, self._BN_MOVING_MEANS = [], []

        mean = tf.reduce_mean(x, axis=0)
        variance = tf.reduce_mean(tf.square(x - mean), axis=0)

        if is_training:
            x_hat = (x - mean) * 1.0 / tf.sqrt(variance + eps)

        else:
            x_hat = (x - self._BN_MOVING_MEANS[scope_name]) * 1.0 / tf.sqrt(self._BN_MOVING_VARS[scope_name] + eps)

        out = gamma * x_hat + beta

        if is_training:

            if scope_name >= len(self._BN_MOVING_MEANS):
                self._BN_MOVING_MEANS.append(mean)

            else:
                self._BN_MOVING_MEANS[scope_name] = self._BN_MOVING_MEANS[scope_name] * momentum + mean * (
                            1.0 - momentum)

            if scope_name >= len(self._BN_MOVING_VARS):
                self._BN_MOVING_VARS.append(variance)
            else:
                self._BN_MOVING_VARS[scope_name] = self._BN_MOVING_VARS[scope_name] * momentum + variance * (
                            1.0 - momentum)

        if debug:
            print('== info start ==')
            print('scope_name = {}'.format(scope_name))
            print('mean = {}'.format(mean))
            print('var = {}'.format(variance))
            print('_BN_MOVING_MEANS = {}'.format(self._BN_MOVING_MEANS[scope_name]))
            print('_BN_MOVING_VARS = {}'.format(self._BN_MOVING_VARS[scope_name]))
            print('output = {}'.format(out))
            print('== info end ==')

        return out


    def Dropout(self,weights,dropProb):
        if dropProb == 1:
            return tf.zeros_like(weights)

        if dropProb == 0:
            return weights

        mask = tf.random.uniform(shape=tf.shape(weights),minval=0.0,maxval=1.0) < 1 - dropProb

        return tf.cast(mask, dtype=tf.float32) * weights / (1.0 - dropProb)

    def moveOneStep(self):
        for dIndex, (dParameter, dVelocity) in enumerate(zip(self.parameters, self.velocity)):
            for elIndex, (elP, elV) in enumerate(zip(dParameter, dVelocity)):
                self.parameters[dIndex][elIndex] = (elP + self.momentum * elV)


    def feedForward(self,data,training=True):

        previousOutput = data

        batchIndex = 0
        for layer, (currentLayer, currentActivation, bias, dropoutLayer, dropoutProbLayer, batchNormLayer) in \
                enumerate(zip(self.Layers, self.activations, self.biases,
                              self.dropout, self.dropoutProb, self.batchNormalization)):

            output = tf.tensordot(previousOutput, currentLayer, axes=1)
            output = output + bias


            if batchNormLayer:
                gamma = self.BatchLayersGamma[batchIndex]
                beta = self.BatchLayersBeta[batchIndex]
                output = self.batch_norm(output, gamma, beta, batchIndex,is_training=training)
                batchIndex += 1


            if currentActivation == "sigmoid":
                output = tf.sigmoid(output)

            elif currentActivation == "tanh":
                output = tf.tanh(output)

            elif currentActivation == "relu":
                output = tf.keras.activations.relu(output)

            elif currentActivation == "selu":
                output = tf.keras.activations.selu(output)

            elif currentActivation == "softmax":
                output = tf.keras.activations.softmax(output)

            elif currentActivation == "linear":
                pass

            else:
                pass


            if dropoutLayer and not training:
                output = self.Dropout(output, dropoutProbLayer)

            previousOutput = output

        return output

    def loss(self,targets,predTargets):

        #REGRESSION
        if self.lossFunction == "MAE":
            error = tf.reduce_mean(tf.abs(targets - predTargets))

        elif self.lossFunction == "MSE":
            error = tf.reduce_mean(tf.square(targets - predTargets))

        elif self.lossFunction == "RMSE":
            error = tf.sqrt(tf.reduce_mean(tf.square(targets - predTargets)))

        elif self.lossFunction == "MSLE":
            error = tf.reduce_mean(tf.square(tf.math.log(targets) - tf.math.log(predTargets)))

        elif self.lossFunction == "Huber":
            huber = tf.keras.losses.Huber(delta=self.delta)
            error = huber(targets,predTargets)

        elif self.lossFunction == "LogCosh":
            LogCosh = tf.keras.LogCosh()
            error = LogCosh(targets,predTargets)


        #BINARY CLASSIFICATION
        elif self.lossFunction == "BCE":
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            error = bce(targets, predTargets)

        #MULTIPLE CLASSIFICATION
        elif self.lossFunction == "CCE":
            cce = tf.keras.losses.CategoricalCrossentropy()
            error = cce(targets, predTargets)

        l1cost = 0.0
        for weights, regLayer in zip(self.Layers, self.l1Regularization):
            if regLayer:
                l1cost += self.Lambda * tf.reduce_sum(tf.abs(weights)) / 2

        l2cost = 0.0
        for weights, regLayer in zip(self.Layers, self.l2Regularization):
            if regLayer:
                l2cost += self.Lambda * tf.reduce_sum(tf.square(weights)) / 2

        error += l1cost + l2cost
        return error

    def backPropagation(self,error,t,lr):
        dError_dWeights = t.gradient(error, self.parameters)
        if not self.optimizer:

            if self.epoch == 0:

                self.velocity = []


                for dIndex,dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    for elP in dParameter:
                        self.velocity[dIndex].append(-lr * elP)


            else:
                for dIndex, dParameter in enumerate(dError_dWeights):
                    for elIndex, elP in enumerate(dParameter):
                        self.velocity[dIndex][elIndex] = - lr * elP


        elif self.optimizer == "momentum" or self.optimizer == "nesterov":


            if self.epoch == 0:

                self.velocity = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    for el in dParameter:
                        self.velocity[dIndex].append(-lr * el)



            else:

                for dIndex, (dParameter, dVelocity) in enumerate(zip(dError_dWeights, self.velocity)):
                    for elIndex, (elP, elV) in enumerate(zip(dParameter, dVelocity)):
                        self.velocity[dIndex][elIndex] = self.momentum * elV - lr * elP



        elif self.optimizer == "adagrad":
            zero = 1e-8

            if self.epoch == 0:

                self.velocity = []
                self.sumGrad = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    self.sumGrad.append([])
                    for elIndex, el in enumerate(dParameter):
                        self.sumGrad[dIndex].append(el * el)
                        self.velocity[dIndex].append(-lr * el / tf.sqrt(self.sumGrad[dIndex][elIndex] + zero))


            else:

                for dIndex, (dParameter, dVelocity, dSG) in enumerate(
                        zip(dError_dWeights, self.velocity, self.sumGrad)):
                    for elIndex, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
                        self.sumGrad[dIndex][elIndex] += elP * elP
                        self.velocity[dIndex][elIndex] = -lr * elP / tf.sqrt(elSG + zero)


        elif self.optimizer == "RMSProp":

            zero = 1e-8

            if self.epoch == 0:

                self.velocity = []
                self.sumGrad = []

                for dIndex, dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    self.sumGrad.append([])
                    for elIndex, el in enumerate(dParameter):
                        self.sumGrad[dIndex].append((1 - self.decayRate) * el * el)
                        self.velocity[dIndex].append(-lr * el / tf.sqrt(self.sumGrad[dIndex][elIndex] + zero))



            else:

                for dIndex, (dParameter, dVelocity, dSG) in enumerate(
                        zip(dError_dWeights, self.velocity, self.sumGrad)):
                    for elIndex, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
                        self.sumGrad[dIndex][elIndex] = self.decayRate * elSG + (1 - self.decayRate) * elP * elP
                        self.velocity[dIndex][elIndex] = -lr * elP / tf.sqrt(elSG + zero)

        elif self.optimizer == "adam" or self.optimizer == "nadam":

            zero = 1e-8


            if self.epoch == 0:

                self.velocity = []
                self.sumMom = []
                self.sumMomHat = []
                self.sumGrad = []
                self.sumGradHat = []


                for dIndex,dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    self.sumGrad.append([])
                    self.sumGradHat.append([])
                    self.sumMom.append([])
                    self.sumMomHat.append([])

                    for el in dParameter:
                        self.sumMom[dIndex].append(-(1 - self.beta1) * el)
                        self.sumMomHat[dIndex].append(el)
                        self.sumGrad[dIndex].append((1 - self.beta2) * el * el)
                        self.sumGradHat[dIndex].append(el * el)
                        self.velocity[dIndex].append(lr * el / tf.sqrt(el * el + zero))

                    dIndex += 1

            else:
                for dIndex, (dParameter, dVelocity, dSG, dSM) in enumerate(
                        zip(dError_dWeights, self.velocity, self.sumGrad, self.sumMom)):
                    for index, (elP, elV, elSG, elSM) in enumerate(zip(dParameter, dVelocity, dSG, dSM)):
                        self.sumMom[dIndex][index] = self.beta1 * elSM - (1 - self.beta1) * elP
                        self.sumGrad[dIndex][index] = self.beta2 * elSG + (1 - self.beta2) * elP * elP
                        self.sumMomHat[dIndex][index] = self.sumMom[dIndex][index] / (
                                    1 - self.beta1 ** (self.epoch + 1))
                        self.sumGradHat[dIndex][index] = self.sumGrad[dIndex][index] / (
                                    1 - self.beta1 ** (self.epoch + 1))
                        self.velocity[dIndex][index] = lr * self.sumMomHat[dIndex][index] / tf.sqrt(
                            self.sumGradHat[dIndex][index] + zero)



        else:


            if self.epoch == 0:

                self.velocity = []


                for dIndex,dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    for elP in dParameter:
                        self.velocity[dIndex].append(-lr * elP)


            else:
                for dIndex, dParameter in enumerate(dError_dWeights):
                    for elIndex, elP in enumerate(dParameter):
                        self.velocity[dIndex][elIndex] = - lr * elP

        if self.gradientClipping:
            norm = tf.norm(self.velocity)
            if norm > self.GCT:
                self.velocity *= self.GCT / norm

    def update_parameters(self):

        for index in range(len(self.Layers)):
            self.Layers[index] += self.velocity[0][index]

        for index in range(len(self.biases)):
            self.biases[index] += self.velocity[3][index]

        for index in range(len(self.BatchLayersGamma)):
            if self.batchNormalization[index]:
                self.BatchLayersGamma[index] += self.velocity[1][index]

        for index in range(len(self.BatchLayersBeta)):

            self.BatchLayersBeta[index] += self.velocity[2][index]

        self.parameters = [self.Layers, self.BatchLayersGamma, self.BatchLayersBeta, self.biases]



    def train(self,lr=0.01,epochs=1500):



        for epoch in range(epochs):
            start=0
            end=self.batch_size

            while start<len(self.x):
                if end>len(self.x):
                    end = len(self.x)


                with tf.GradientTape() as t:


                    if self.optimizer=="nesterov" or self.optimizer=="nadam":
                        if self.epoch!=0:
                            self.moveOneStep()


                    t.watch(self.parameters)

                    data = self.x[start:end]
                    targets = self.y[start:end]


                    predTargets = self.feedForward(data)


                    error = self.loss(targets,predTargets)

                    if self.epoch%100==0:
                        print(self.epoch)
                        print(error)

                    self.backPropagation(error,t,lr)


                    self.update_parameters()


                start += self.batch_size
                end += self.batch_size

            self.epoch += 1

        return error

    def predict(self,x):

        if len(x.shape)==1:
            data = to_tensor(np.reshape(x,(-1,1)))

        else:
            data = to_tensor(x)



        pred = self.feedForward(data,training = False)

        return pred.numpy()


    def plotPredictions(self,axes=None,lr=0.01,epochs=1000, typeOfProblem="Classification",
                        offset=0.3,subplots=False,ax=None,fig=None,name = ""):


        try:
            self.firstPlot
        except:
            self.firstPlot = True



        if not subplots:
            plt.rcParams["figure.figsize"] = [10., 8.]
            plt.rcParams["figure.autolayout"] = True

        if typeOfProblem == "regression2D":

            if not subplots:
                fig, ax = plt.subplots()


            if self.firstPlot:


                if not axes:
                    maxX = tf.reduce_max(self.x)
                    minX = tf.reduce_min(self.x)
                    disX = maxX - minX
                    maxX += disX * offset
                    minX -= disX * offset
                    maxY = tf.reduce_max(self.y)
                    minY = tf.reduce_min(self.y)
                    disY = maxY - minY
                    maxY += disY * offset
                    minY -= disY * offset

                    self.axes = [minX, maxX, minY, maxY]

                else:
                    self.axes = axes

                self.plotData = np.linspace(self.axes[0], self.axes[1], 100)

                if self.produceData:
                    self.plotData = self.manipulate_data(self.plotData, columns=self.produceData[0], func=self.produceData[1])



            if not subplots:

                def animate(i):

                    ax.clear()

                    ax.set_xlim([self.axes[0], self.axes[1]])
                    ax.set_ylim([self.axes[2], self.axes[3]])

                    error = self.train(lr, epochs=1)

                    y_pred = self.predict(self.plotData).reshape(self.plotData.shape)

                    plt.title("epoch:" + str(self.epoch) + " Error:" + str(error.numpy()))

                    ax.scatter(self.x, self.y, alpha=1, s=3)


                    ax.plot(self.plotData,y_pred,'--r','LineWidth',0.3)


                ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)
                plt.show()

            else:
                ax.set_xlim([self.axes[0], self.axes[1]])
                ax.set_ylim([self.axes[2], self.axes[3]])

                error = self.train(lr, epochs=1)

                y_pred = self.predict(self.plotData).reshape(self.plotData.shape)

                plt.title("epoch:" + str(self.epoch) + " Error:" + str(error.numpy()))

                ax.scatter(self.x, self.y, alpha=1, s=3)

                ax.plot(self.plotData, y_pred, '--r', 'LineWidth', 0.3)

        elif typeOfProblem == "regression3D":

            if not subplots:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')


            if self.firstPlot:
                self.x1 = extract(self.x, 0)
                self.x2 = extract(self.x, 1)

                maxZ = tf.reduce_max(self.y)
                minZ = tf.reduce_min(self.y)
                disZ = maxZ - minZ
                maxZ += disZ * offset
                minZ -= disZ * offset



                if not axes:
                    maxX = tf.reduce_max(self.x1)
                    minX = tf.reduce_min(self.x1)
                    disX = maxX - minX
                    maxX += disX * offset
                    minX -= disX * offset
                    maxY = tf.reduce_max(self.x2)
                    minY = tf.reduce_min(self.x2)
                    disY = maxY - minY
                    maxY += disY * offset
                    minY -= disY * offset
                    self.axes = [minX, maxX, minY, maxY, minZ, maxZ]

                else:
                    self.axes = axes + [minZ,maxZ]


                self.X = np.linspace(self.axes[0], self.axes[1], 100)

                self.Y = np.linspace(self.axes[2], self.axes[3], 100)

                self.X,self.Y = np.meshgrid(self.X, self.Y)

                self.plotData = np.c_[self.X.ravel(), self.Y.ravel()]

                if self.produceData:
                    self.plotData = self.manipulate_data(self.plotData, columns=self.produceData[0], func=self.produceData[1])

            if not subplots:

                def animate(i):
                    ax.clear()

                    ax.set_xlim([self.axes[0], self.axes[1]])
                    ax.set_ylim([self.axes[2], self.axes[3]])
                    ax.set_zlim([self.axes[4], self.axes[5]])


                    error = self.train(lr, epochs=1)


                    y_pred = self.predict(self.plotData).reshape(self.X.shape)

                    plt.title("epoch:" + str(self.epoch) + " Error:" + str(error.numpy()))


                    norm = plt.Normalize(y_pred.min(), y_pred.max())
                    rcolors = cm.viridis(norm(y_pred))
                    rcount, ccount, _ = rcolors.shape

                    surf = ax.plot_surface(self.X, self.Y, y_pred, rcount=rcount, ccount=ccount, facecolors=rcolors, shade=False)
                    surf.set_facecolor((0, 0, 0, 0))


                    ax.scatter(self.x1, self.x2, self.y,c=self.y, alpha=1, s=5,depthshade=True,cmap=plt.get_cmap("RdYlBu_r"))


                ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)
                plt.show()

            else:
                ax.set_xlim([self.axes[0], self.axes[1]])
                ax.set_ylim([self.axes[2], self.axes[3]])
                ax.set_zlim([self.axes[4], self.axes[5]])

                error = self.train(lr, epochs=1)

                y_pred = self.predict(self.plotData).reshape(self.X.shape)

                ax.set_title("NN " + name + ", Error " + str(error.numpy()))
                ax.plot_wireframe(self.X, self.Y, y_pred, rstride=1, cstride=1, linewidth=0.004, antialiased=False,
                                  alpha=0.8, cmap=plt.get_cmap("RdYlBu_r"))
                ax.scatter(self.x1, self.x2, self.y, color='red', alpha=1, s=5, depthshade=True)





        elif typeOfProblem=="classification":

            if not subplots:
                fig, ax = plt.subplots()


            if self.firstPlot:

                self.x1 = extract(self.x, 0)
                self.x2 = extract(self.x, 1)


                if not axes:
                    maxX = tf.reduce_max(self.x1)
                    minX = tf.reduce_min(self.x1)
                    disX = maxX - minX
                    maxX += disX/2
                    minX -= disX/2
                    maxY = tf.reduce_max(self.x2)
                    minY = tf.reduce_min(self.x2)
                    disY = maxY - minY
                    maxY += disY / 2
                    minY -= disY / 2


                    self.axes = [minX, maxX, minY, maxY]


                else:
                    self.axes = axes


                self.X = np.linspace(self.axes[0],self.axes[1],100)

                self.Y = np.linspace(self.axes[2],self.axes[3],100)

                self.X,self.Y = np.meshgrid(self.X,self.Y)

                self.plotData = np.c_[self.X.ravel(),self.Y.ravel()]




                if self.produceData:
                    self.plotData = self.manipulate_data(self.plotData, columns=self.produceData[0], func=self.produceData[1])

            if not subplots:
                def animate(i):
                    ax.clear()
                    ax.set_xlim([self.axes[0], self.axes[1]])
                    ax.set_ylim([self.axes[2], self.axes[3]])

                    error = self.train(lr,epochs=1)

                    y_pred = self.predict(self.plotData).reshape(x.shape)

                    plt.title("epoch:"+str(self.epoch)+" Error:"+str(error.numpy()))
                    ax.contourf(self.X, self.Y, y_pred, cmap=plt.get_cmap("RdYlBu_r"), alpha=0.75)
                    ax.scatter(self.x1, self.x2, alpha=1, c=self.y, s=3, cmap=plt.get_cmap('viridis'))


                ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)
                plt.show()

            else:
                ax.set_xlim([self.axes[0], self.axes[1]])
                ax.set_ylim([self.axes[2], self.axes[3]])

                error = self.train(lr, epochs=1)

                y_pred = self.predict(self.plotData).reshape(self.X.shape)

                ax.set_title("NN "+ name + ", Error " + str(error.numpy()))
                ax.contourf(self.X, self.Y, y_pred, cmap=plt.get_cmap("RdYlBu_r"), alpha=0.75)
                ax.scatter(self.x1, self.x2, alpha=1, c=self.y, s=3, cmap=plt.get_cmap('viridis'))



        elif typeOfProblem=="classification3":

            if not subplots:
                fig, ax = plt.subplots()


            if self.firstPlot:

                self.x1 = extract(self.x, 0)
                self.x2 = extract(self.x, 1)


                if not axes:
                    maxX = tf.reduce_max(self.x1)
                    minX = tf.reduce_min(self.x1)
                    disX = maxX - minX
                    maxX += disX/2
                    minX -= disX/2
                    maxY = tf.reduce_max(self.x2)
                    minY = tf.reduce_min(self.x2)
                    disY = maxY - minY
                    maxY += disY / 2
                    minY -= disY / 2


                    self.axes = [minX, maxX, minY, maxY]


                else:
                    self.axes = axes


                self.X = np.linspace(self.axes[0],self.axes[1],100)

                self.Y = np.linspace(self.axes[2],self.axes[3],100)

                self.X,self.Y = np.meshgrid(self.X,self.Y)

                self.plotData = np.c_[self.X.ravel(),self.Y.ravel()]




                if self.produceData:
                    self.plotData = self.manipulate_data(self.plotData, columns=self.produceData[0], func=self.produceData[1])

                self.groupsx1 = [[] for _ in range(self.y.shape[-1])]
                self.groupsx2 = [[] for _ in range(self.y.shape[-1])]
                filter = np.array([i for i in range(self.y.shape[-1])])

                for dx1,dx2,dy in zip(self.x1,self.x2,self.y):
                    index = np.sum(dy*filter).astype(np.int32)
                    self.groupsx1[index].append(dx1)
                    self.groupsx2[index].append(dx2)


                from matplotlib.colors import ListedColormap
                self.custom_cmap = ListedColormap(listOfColors[:self.y.shape[-1]])

                self.colors = listOfColors[:self.y.shape[-1]]
                self.names = np.array([str(i) for i in range(self.y.shape[-1])])

            if not subplots:

                def animate(i):
                    ax.clear()


                    ax.set_xlim([self.axes[0], self.axes[1]])
                    ax.set_ylim([self.axes[2], self.axes[3]])

                    error = self.train(lr,epochs=1)


                    y_pred = self.predict(self.plotData)
                    y_pred = np.argmax(y_pred, axis=-1)
                    y_pred = y_pred.reshape(self.X.shape)




                    plt.title("epoch:"+str(self.epoch)+" Error:"+str(error.numpy()))

                    ax.contourf(self.X, self.Y, y_pred, cmap=self.custom_cmap, alpha=0.4)

                    for name,groupx1,groupx2,color in zip(self.names,self.groupsx1,self.groupsx2,self.colors):

                        ax.scatter(groupx1, groupx2, alpha=1, s=3, color=color,label=name)


                ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)
                plt.show()

            else:
                ax.set_xlim([self.axes[0], self.axes[1]])
                ax.set_ylim([self.axes[2], self.axes[3]])

                error = self.train(lr, epochs=1)

                y_pred = self.predict(self.plotData)
                y_pred = np.argmax(y_pred, axis=-1)
                y_pred = y_pred.reshape(self.X.shape)


                ax.set_title("NN "+ name + ", Error " + str(error.numpy()))
                ax.contourf(self.X, self.Y, y_pred, cmap=self.custom_cmap, alpha=0.4)

                for name, groupx1, groupx2, color in zip(self.names, self.groupsx1, self.groupsx2, self.colors):
                    ax.scatter(groupx1, groupx2, alpha=1, s=3, color=color, label=name)

        self.firstPlot = False


    def save(self,name):
        self.name = name
        file = open(self.name+'.txt','wb')
        pickle.dump(self, file)

    def load(self,name):
        out = loadNN(name)
        self.__dict__.update(out.__dict__)



def loadNN(name):
    file = open(name + ".txt", 'rb')
    out = pickle.load(file)
    return copy.deepcopy(out)




def plotMultipleNNs(NNs,axes=None,lr=0.01,epochs=1000,typeOfProblem="classification3",offset=0.3,manipulateData=None,height=2,length=3,names=None):
    plots = len(NNs)

    plt.rcParams["figure.figsize"] = [13., 8.]
    plt.rcParams["figure.autolayout"] = True

    if not names:
        names = [str(i) for i in range(len(NNs))]

    if typeOfProblem == "regression3D":
        fig = plt.figure()

    else:
        fig, axs = plt.subplots(height,length,figsize=(16,16))




    def animate(i):
        if typeOfProblem=="regression3D":
            fig.clear(True)

        fig.suptitle("epoch: " + str(i),fontsize=20)

        for index,(name,NN) in enumerate(zip(names,NNs)):



            if height==1:
                if typeOfProblem == "regression3D":

                    ax = fig.add_subplot(1, 2, index+1, projection='3d')

                else:
                    axs[index].clear()
                    ax = axs[index]

                NN.plotPredictions(axes=axes,lr=lr,epochs=epochs,
                                   typeOfProblem=typeOfProblem,offset=offset,subplots=True,ax=ax,fig=fig, name=name)


            else:

                if typeOfProblem == "regression3D":
                    ax = fig.add_subplot(2, index // length+1, index % height+1, projection='3d')

                else:
                    axs[index // length, index % height].clear()
                    ax = axs[index // length, index % height]

                NN.plotPredictions(axes=axes, lr=lr, epochs=epochs,
                                   typeOfProblem=typeOfProblem, offset=offset, subplots=True,
                                   ax=ax, fig=fig, name=name)

            if not typeOfProblem == "regression3D":
                for ax in axs.flat:
                    ax.set(xlabel='x-label', ylabel='y-label')


                for ax in axs.flat:
                    ax.label_outer()

    ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)

    plt.show()



def extract(lst,pos):
    return [element[pos] for element in lst]

def oneHot(targets):
    targets
    y = np.zeros((targets.size, targets.max() + 1))
    y[np.arange(targets.size), targets] = 1
    return y.astype(int)

#

# w = tf.constant(1.)
# b = tf.constant(0.)
# center = tf.constant(2.)
#
# u, v = np.mgrid[0:2*np.pi:40j, 0:2*np.pi:40j]
# x = np.cos(u)*np.sin(v)
# x=x.flatten()
# y = np.sin(u)*np.sin(v)
# y=y.flatten()
# z = np.cos(v)
# z=z.flatten()
#
# indices = [i for i,dz in enumerate(z) if dz>=0]
# x,y,z = x[indices],y[indices],z[indices]
#
#
#
# data = np.vstack((x,y)).T
# targets = z
#
#
# targets += tf.random.normal(shape=y.shape, stddev=0.1)
#
#
# Layers = [10,10,10,10,10,10]
# B = NN(data,
#        targets,
#        Layers,
#        loss_function="MSE",
#        activations=["selu","selu","selu","selu","selu","selu","linear"],
#        initializations="lecun_normal",
#        batch_normalization=True,
#        optimizer="nesterov",
#        shuffle=True,
#        l2Regularization=True
#        )
#
# K = NN(data,
#        targets,
#        Layers,
#        loss_function="MSE",
#        activations=["selu","selu","selu","selu","selu","selu","linear"],
#        initializations="lecun_normal",
#        batch_normalization=True,
#        optimizer="nesterov",
#        shuffle=True,
#        l2Regularization=True
#        )
#
# K.plotPredictions(typeOfProblem="regression3D")
#
# plotMultipleNNs([B,K],typeOfProblem="regression3D",height=1,length=2)
#
# xXOR = np.array([[0,0],[0,1],[1,0],[1,1]])
#
# yXOR = np.array([[0],[1],[1],[0]])
#
#
# Layers = [4,4]
# D = NN(xXOR,
#        yXOR,
#        Layers,
#        loss_function="BCE",
#        activations=["tanh","tanh","sigmoid"],
#        initializations="xavier_uniform",
#        batch_normalization=False,
#        optimizer="nesterov",
#        batch_size=None,
#        shuffle=True
#        )
#
#
# D.plotPredictions(typeOfProblem="classification",lr=0.1)
#
#
#
# x = sklearn.datasets.make_circles(n_samples=300,noise=0.15,factor=0.2)
#
# targets = x[1]
# data = x[0]
#
# Layers = [5,5,5,5]
# C = NN(data,
#        targets,
#        Layers,
#        loss_function="BCE",
#        activations=["selu","selu","selu","selu","sigmoid"],
#        initializations=["lecun_uniform","lecun_uniform","lecun_uniform","lecun_uniform","xavier_uniform"],
#        batch_normalization=True,
#        optimizer="nadam",
#        batch_size=None,
#        shuffle=True
#        )
#
# L = NN(data,
#        targets,
#        Layers,
#        loss_function="BCE",
#        activations=["selu","selu","selu","selu","sigmoid"],
#        initializations=["lecun_uniform","lecun_uniform","lecun_uniform","lecun_uniform","xavier_uniform"],
#        batch_normalization=True,
#        optimizer="nesterov",
#        batch_size=None,
#        shuffle=True
#        )
#
#
# plotMultipleNNs([C,L],typeOfProblem="classification",height=1,length=2)

#
#
#
# w = tf.constant(10.)
# b = tf.constant(2.)
#
# x = tf.random.uniform([200],minval=-10.,maxval=10.)
# y = tf.cos(x)*w + b
#
# Layers = [5,5,5,5]
# A = NN(x,
#        y,
#        Layers,
#        loss_function="MSE",
#        activations=["selu","selu","selu","selu","linear"],
#        initializations="lecun_normal",
#        batch_normalization=True,
#        optimizer="nadam",
#        batch_size=None
#        )
#
# V = NN(x,
#        y,
#        Layers,
#        loss_function="MSE",
#        activations=["selu","selu","selu","selu","linear"],
#        initializations="lecun_normal",
#        batch_normalization=True,
#        optimizer="nadam",
#        batch_size=None
#        )

# plotMultipleNNs([A,V],height=1,length=2,typeOfProblem="regression2D")
# A.plotPredictions(typeOfProblem="regression2D")
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#
# from sklearn.datasets import make_blobs
# # from sklearn.datasets import make_classification
#
# x , y = make_blobs(n_samples=300, centers=4)
# # x,y = make_classification(n_samples=300, n_features=2, n_informative=2,n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2,class_sep=1,flip_y=0.2)
#
# y = oneHot(y)
#
#
# Layers = [10,10,10,10]
# F = NN(x,
#        y,
#        Layers,
#        loss_function="CCE",
#        activations=["selu","selu","selu","selu","softmax"],
#        initializations="lecun_normal",
#        batch_normalization=True,
#        optimizer="nesterov"
#        )
#
# Layers = [10,10,10,10]
# G = NN(x,
#        y,
#        Layers,
#        loss_function="CCE",
#        activations=["selu","selu","selu","selu","softmax"],
#        initializations="lecun_normal",
#        batch_normalization=True,
#        optimizer="nadam"
#        )
#
#
#
# plotMultipleNNs([F,G],typeOfProblem="classification3",height=1,length=2,names=["nesterov","nadam"])
#





