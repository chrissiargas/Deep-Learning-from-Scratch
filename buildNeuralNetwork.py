import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import sklearn.datasets
import copy
import pickle

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "svm"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

tf.config.optimizer.set_jit(True)


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

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*sigmoid(-x)

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
                 dropoutProb=0.5):

        self.numberOfLayers = len(layers) + 1

        if len(data.shape)==1:
            data = np.reshape(data,(-1,1))

        self.manipulateData = manipulate

        if self.manipulateData:
            self.x = to_tensor(self.manipulate_data(data,columns=manipulate[0],func=manipulate[1]))

        else:
            self.x = to_tensor(data)


        if len(targets.shape)==1:
            targets = np.reshape(targets,(-1,1))

        self.y = to_tensor(targets)




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
                newLayer = rng.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)

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


    def manipulate_data(x, columns=None, func=None):
        data = []

        if columns == None or func == None:
            return

        for index in range(len(x)):

            data.append(np.append(x[index], func(x[index][columns])))

        data = np.array(data)
        return data


    def batch_norm(self, x, gamma, beta, scope_name, is_training=True, debug=False):
        eps = 1e-5
        momentum = self.batch_norm_momentum

        try:
            self._BN_MOVING_VARS, self._BN_MOVING_MEANS
        except:
            self._BN_MOVING_VARS, self._BN_MOVING_MEANS = {}, {}

        mean = tf.reduce_mean(x, axis=0)
        variance = tf.reduce_mean((x - mean) ** 2, axis=0)

        if is_training:
            x_hat = (x - mean) * 1.0 / tf.sqrt(variance + eps)

        else:
            x_hat = (x - self._BN_MOVING_MEANS[scope_name]) * 1.0 / tf.sqrt(self._BN_MOVING_VARS[scope_name] + eps)

        out = gamma * x_hat + beta

        if is_training:
            if scope_name not in self._BN_MOVING_MEANS:
                self._BN_MOVING_MEANS[scope_name] = mean
            else:
                self._BN_MOVING_MEANS[scope_name] = self._BN_MOVING_MEANS[scope_name] * momentum + mean * (1.0 - momentum)
            if scope_name not in self._BN_MOVING_VARS:
                self._BN_MOVING_VARS[scope_name] = variance
            else:
                self._BN_MOVING_VARS[scope_name] = self._BN_MOVING_VARS[scope_name] * momentum + variance * (1.0 - momentum)

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
                output = self.batch_norm(output, gamma, beta, layer,is_training=training)
                batchIndex += 1


            if currentActivation == "sigmoid":
                output = tf.sigmoid(output)

            elif currentActivation == "tanh":
                output = tf.tanh(output)

            elif currentActivation == "relu":
                output = tf.keras.activations.relu(output)

            elif currentActivation == "linear":
                pass

            else:
                output = tf.sigmoid(output)


            if dropoutLayer and not training:
                output = self.Dropout(output, dropoutProbLayer)

            previousOutput = output

        return output

    def loss(self,targets,predTargets):

        if self.lossFunction == "MSE":
            error = tf.reduce_sum(tf.square(targets - predTargets))

        elif self.lossFunction == "BCE":
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            error = bce(targets, predTargets)

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

        if not self.optimizer:
            dError_dWeights = t.gradient(error, self.parameters)

            if self.epoch == 0:

                self.velocity = []

                dPindex = 0
                for dParameter in dError_dWeights:

                    self.velocity.append([])
                    for elP in dParameter:
                        self.velocity[dPindex].append(-lr * elP)
                    dPindex += 1

            else:
                for dPindex, dParameter in enumerate(dError_dWeights):
                    for index, elP in enumerate(dParameter):
                        self.velocity[dPindex][index] = - lr * elP


        elif self.optimizer == "momentum" or self.optimizer == "nesterov":
            dError_dWeights = t.gradient(error, self.parameters)

            if self.epoch == 0:

                self.velocity = []

                for dPindex, dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    for el in dParameter:
                        self.velocity[dPindex].append(-lr * el)



            else:

                for dPindex, (dParameter, dVelocity) in enumerate(zip(dError_dWeights, self.velocity)):
                    for index, (elP, elV) in enumerate(zip(dParameter, dVelocity)):
                        self.velocity[dPindex][index] = self.momentum * elV - lr * elP



        elif self.optimizer == "adagrad":
            zero = 1e-8
            dError_dWeights = t.gradient(error, self.parameters)

            if self.epoch == 0:

                self.velocity = []
                self.sumGrad = []

                for dPindex, dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    self.sumGrad.append([])
                    for index, el in enumerate(dParameter):
                        self.sumGrad[dPindex].append(el * el)
                        self.velocity[dPindex].append(-lr * el / tf.sqrt(self.sumGrad[dPindex][index] + zero))


            else:

                for dPindex, (dParameter, dVelocity, dSG) in enumerate(
                        zip(dError_dWeights, self.velocity, self.sumGrad)):
                    for index, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
                        self.sumGrad[dPindex][index] += elP * elP
                        self.velocity[dPindex][index] = -lr * elP / tf.sqrt(elSG + zero)


        elif self.optimizer == "RMSProp":

            zero = 1e-8
            dError_dWeights = t.gradient(error, self.parameters)

            if self.epoch == 0:

                self.velocity = []
                self.sumGrad = []

                for dPindex, dParameter in enumerate(dError_dWeights):

                    self.velocity.append([])
                    self.sumGrad.append([])
                    for index, el in enumerate(dParameter):
                        self.sumGrad[dPindex].append((1 - self.decayRate) * el * el)
                        self.velocity[dPindex].append(-lr * el / tf.sqrt(self.sumGrad[dPindex][index] + zero))



            else:

                for dPindex, (dParameter, dVelocity, dSG) in enumerate(
                        zip(dError_dWeights, self.velocity, self.sumGrad)):
                    for index, (elP, elV, elSG) in enumerate(zip(dParameter, dVelocity, dSG)):
                        self.sumGrad[dPindex][index] = self.decayRate * elSG + (1 - self.decayRate) * elP * elP
                        self.velocity[dPindex][index] = -lr * elP / tf.sqrt(elSG + zero)

        elif self.optimizer == "adam" or self.optimizer == "nadam":

            zero = 1e-8
            dError_dWeights = t.gradient(error, self.parameters)

            if self.epoch == 0:

                self.velocity = []
                self.sumMom = []
                self.sumMomHat = []
                self.sumGrad = []
                self.sumGradHat = []

                dPindex = 0
                for dParameter in dError_dWeights:

                    self.velocity.append([])
                    self.sumGrad.append([])
                    self.sumGradHat.append([])
                    self.sumMom.append([])
                    self.sumMomHat.append([])

                    for el in dParameter:
                        self.sumMom[dPindex].append(-(1 - self.beta1) * el)
                        self.sumMomHat[dPindex].append(el)
                        self.sumGrad[dPindex].append((1 - self.beta2) * el * el)
                        self.sumGradHat[dPindex].append(el * el)
                        self.velocity[dPindex].append(lr * el / tf.sqrt(el * el + zero))

                    dPindex += 1

            else:
                for dPindex, (dParameter, dVelocity, dSG, dSM) in enumerate(
                        zip(dError_dWeights, self.velocity, self.sumGrad, self.sumMom)):
                    for index, (elP, elV, elSG, elSM) in enumerate(zip(dParameter, dVelocity, dSG, dSM)):
                        self.sumMom[dPindex][index] = self.beta1 * elSM - (1 - self.beta1) * elP
                        self.sumGrad[dPindex][index] = self.beta2 * elSG + (1 - self.beta2) * elP * elP
                        self.sumMomHat[dPindex][index] = self.sumMom[dPindex][index] / (
                                    1 - self.beta1 ** (self.epoch + 1))
                        self.sumGradHat[dPindex][index] = self.sumGrad[dPindex][index] / (
                                    1 - self.beta1 ** (self.epoch + 1))
                        self.velocity[dPindex][index] = lr * self.sumMomHat[dPindex][index] / tf.sqrt(
                            self.sumGradHat[dPindex][index] + zero)



        else:
            dError_dWeights = t.gradient(error, self.parameters)

            if self.epoch == 0:

                self.velocity = []

                dPindex = 0
                for dParameter in dError_dWeights:

                    self.velocity.append([])
                    for elP in dParameter:
                        self.velocity[dPindex].append(-lr * elP)
                    dPindex += 1

            else:
                for dPindex, dParameter in enumerate(dError_dWeights):
                    for index, elP in enumerate(dParameter):
                        self.velocity[dPindex][index] = - lr * elP

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
                    # print(output)


                    error = self.loss(targets,predTargets)


                    self.backPropagation(error,t,lr)


                    self.update_parameters()


                start += self.batch_size
                end += self.batch_size

            self.epoch += 1

        return error

    def predict(self,x):
        data = to_tensor(x)

        pred = self.feedForward(data,training = False)

        return pred.numpy()

    def plotPredictions(self,axes=[-1.5,1.5,-1.5,1.5],lr=0.01,epochs=1000):
        x = np.linspace(axes[0],axes[1],100)

        y = np.linspace(axes[2],axes[3],100)

        x,y = np.meshgrid(x,y)

        X = np.c_[x.ravel(),y.ravel()]


        fig, ax = plt.subplots()
        x1 = extract(self.x,0)
        x2 = extract(self.x,1)

        if self.manipulateData:
            plotData = self.manipulate_data(X, columns=self.manipulateData[0], func=self.manipulateData[1])

        else:
            plotData = X


        def animate(i):
            ax.clear()

            error = self.train(lr,epochs=1)

            y_pred = self.predict(plotData).reshape(x.shape)

            plt.title("epoch:"+str(self.epoch)+" Error:"+str(error))
            ax.scatter(x1,x2,alpha=1,c=self.y,s=3)
            ax.contourf(x, y, y_pred, cmap=plt.cm.brg, alpha=0.2)

        ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)
        plt.show()



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



def plotMultipleNNs(NNs,data,axes=[-1.5,1.5,-1.5,1.5],lr=0.01,epochs=1000,manipulateData=None,height=2,length=3,names=None):
    plots = len(NNs)

    if not names:
        names = range(len(NNs))

    x = np.linspace(axes[0], axes[1], 100)

    y = np.linspace(axes[2], axes[3], 100)

    x, y = np.meshgrid(x, y)

    X = np.c_[x.ravel(), y.ravel()]


    fig, axs = plt.subplots(height,length,figsize=(16,16))
    x1 = extract(data, 0)
    x2 = extract(data, 1)

    plotData = []
    if manipulateData:

        if len(manipulateData.shape)==1:
            for _ in range(plots):
                plotData.append(NN.manipulate_data(X, columns=manipulateData[0], func=manipulateData[1]))

        elif len(manipulateData.shape[0])==plots:

            for plot in range(plots):
                plotData.append(NN.manipulate_data(X, columns=manipulateData[plot][0], func=manipulateData[plot][1]))
    else:
        for _ in range(plots):
            plotData.append(X)

    def animate(i):


        for index,(name,NN) in enumerate(zip(names,NNs)):

            if height==1:
                axs[index].clear()

                error = NN.train(lr, epochs=1)

                y_pred = NN.predict(plotData[index]).reshape(x.shape)

                axs[index].title.set_text(
                    "NN " + name + ", Epoch " + str(i) + ", Error:" + str(error.numpy()))

                axs[index].contourf(x, y, y_pred, cmap=plt.get_cmap("RdYlBu_r"), alpha=0.75)
                axs[index].scatter(x1, x2, alpha=1, c=NN.y, s=3, cmap=plt.get_cmap('viridis'))

            else:
                axs[index // length,index % height].clear()

                error = NN.train(lr, epochs=1)

                y_pred = NN.predict(plotData[index]).reshape(x.shape)


                axs[index // length,index % height].title.set_text("NN "+ name + ", Epoch " + str(i) + ", Error:" + str(error.numpy()))

                axs[index // length,index % height].contourf(x, y, y_pred, cmap=plt.get_cmap("RdYlBu_r"), alpha=0.75)
                axs[index // length, index % height].scatter(x1, x2, alpha=1, c=NN.y, s=3, cmap=plt.get_cmap('viridis'))
            for ax in axs.flat:
                ax.set(xlabel='x-label', ylabel='y-label')


            for ax in axs.flat:
                ax.label_outer()

    ani = animation.FuncAnimation(fig=fig, func=animate, frames=epochs, interval=1, blit=False, repeat=False)
    plt.show()


def extract(lst,pos):
    return [element[pos] for element in lst]

x = sklearn.datasets.make_circles(n_samples=300,noise=0.15,factor=0.2)

x1 = extract(x[0],0)
x2 = extract(x[0],1)
y = x[1]

# plt.scatter(x1,x2,alpha=1,c=y,s=3)
# plt.show()

xXOR = np.array([[0,0],[0,1],[1,0],[1,1]])

yXOR = np.array([[0],[1],[1],[0]])



Layers = [4,3]

targets = x[1]
data = x[0]

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
A = NN(data,
       targets,
       Layers,
       activations="tanh",
       initializations="xavier_uniform",
       loss_function="BCE",
       batch_normalization=True,
       optimizer="adam")

B = NN(data,
       targets,
       Layers,
       activations="tanh",
       initializations="xavier_uniform",
       loss_function="BCE",
       batch_normalization=False,
       batch_size=300,
       optimizer="adam")




plotMultipleNNs(NNs=[A,B],data=data,height=2,length=2,names=["with BN","without BN"])


# NNs = []
# for _ in range(6):
#     NNs.append(copy.deepcopy(A))

#
# NNs[1].optimizer = "momentum"
# NNs[2].optimizer = "nesterov"
# NNs[3].optimizer = "RMSProp"
# NNs[4].optimizer = "adam"
# NNs[5].optimizer = "nadam"





# print(x[0])
# print(x[1])
# C = NeuralNetwork(xXOR,yXOR)
# C.plotPredictions()
#
# B = NN(xXOR,yXOR,[4],activations="sigmoid",initializations="xavier_normal",)
# B.plotPredictions()


# x = [[1,2],[3,4],[4,6]]
# x = to_tensor(x)
# gamma = to_tensor([1,1])
# beta = to_tensor([0,0])
#
# batch_norm(x,gamma,beta,0,debug=True)






