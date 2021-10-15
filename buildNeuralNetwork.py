import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import sklearn.datasets

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "svm"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def to_tensor(npArray):
    tensor = tf.convert_to_tensor(npArray,dtype=tf.float32)
    return tensor




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
    def __init__(self,data,targets,layers,activations=None,initializations=None,
                 gradientClipping=False,gradientClippingThreshold=1.0,
                 batch_size=None,loss_function="MSE"):

        if len(data.shape)==1:
            data = np.reshape(data,(-1,1))
        self.x = to_tensor(data)


        if len(targets.shape)==1:
            targets = np.reshape(targets,(-1,1))
        self.y = to_tensor(targets)



        if batch_size:
            self.batch_size = batch_size

        else:
            self.batch_size = len(self.x)


        if not activations:
            self.activations = ['sigmoid' for _ in range(len(layers) + 1)]

        elif isinstance(activations, str):
            self.activations = [activations for _ in range(len(layers) + 1)]

        else:
            self.activations = activations


        if not initializations:
            self.initializations = ['xavier_uniform' for _ in range(len(layers)+1)]

        elif isinstance(initializations, str):
            self.initializations = [initializations for _ in range(len(layers) + 1)]

        else:
            self.initializations = initializations

        #exploding/vanishing gradient
        self.gradientClipping = gradientClipping
        self.GCT = gradientClippingThreshold

        self.lossFunction = loss_function

        self.numberOfLayers = len(layers)+1

        self.layerNeurons = layers
        self.layerNeurons.insert(0,data.shape[1])

        self.Layers = []
        nextLayer = targets.shape[1]

        for layer,initializer in zip(self.layerNeurons[::-1],self.initializations[::-1]):

            # exploding/vanishing gradient
            if initializer=="xavier_uniform": #tanh,sigmoid,logistic
                newLayer = tf.random.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)
                newLayer *= tf.sqrt(6/(layer+nextLayer))

            elif initializer=="xavier_normal":
                newLayer = tf.random.normal(shape=[layer, nextLayer], dtype=tf.float32)
                newLayer *= tf.sqrt(2/(layer+nextLayer))

            elif initializer=="he_uniform": #relu and variables
                newLayer = tf.random.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)
                newLayer *= tf.sqrt(6/layer)

            elif initializer=="he_normal":
                newLayer = tf.random.normal(shape=[layer, nextLayer], dtype=tf.float32)
                newLayer *= tf.sqrt(2 / layer)

            elif initializer=="lecun_uniform": #selu
                newLayer = tf.random.uniform(shape=[layer, nextLayer], minval=-1., maxval=1., dtype=tf.float32)
                newLayer *= tf.sqrt(3 / layer)

            elif initializer=="lecun_normal":
                newLayer = tf.random.normal(shape=[layer, nextLayer], dtype=tf.float32)
                newLayer *= tf.sqrt(1 / layer)

            self.Layers.append(newLayer)
            nextLayer = layer


        self.Layers.reverse()

    def __str__(self):
        string = ""
        for index,weights in enumerate(self.Layers):
            string+="Layer:"+str(index+1)+"\n"
            for weight in weights:
                string+="["
                for i,element in enumerate(weight):
                    if i!=0:
                        string+=","
                    string+=str(element.numpy())
                string+="]"+"\n"
            string+="\n\n"

        return string

    def train(self,lr=1,epochs=1500):
        for i in range(epochs):
            start=0
            end=self.batch_size

            while end<=len(self.x):


                with tf.GradientTape() as t:
                    t.watch(self.Layers)

                    previousOutput = self.x[start:end]


                    for currentLayer,currentActivation in zip(self.Layers,self.activations):
                        output = tf.tensordot(previousOutput,currentLayer,axes=1)

                        if currentActivation=="sigmoid":
                            output = tf.sigmoid(output)

                        elif currentActivation=="tanh":
                            output = tf.tanh(output)

                        elif currentActivation=="relu":
                            output = tf.keras.activations.selu(output)

                        elif currentActivation=="linear":
                            pass

                        else:
                            output = tf.sigmoid(output)

                        previousOutput = output

                    # print(output)

                    if self.lossFunction=="MSE":
                        error = tf.reduce_sum(tf.square(self.y[start:end]-output))

                    elif self.lossFunction=="BCE":
                        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                        error = bce(self.y[start:end],output)




                    dError_dWeights = t.gradient(error,self.Layers)
                    # print(dError_dWeights)
                    if self.gradientClipping:
                        norm = tf.norm(dError_dWeights)
                        if norm>self.GCT:
                            dError_dWeights *= self.GCT / norm

                    for index in range(len(self.Layers)):
                        self.Layers[index] -= lr*dError_dWeights[index]



                # print(self)
                # print(error)
                start += self.batch_size
                end += self.batch_size

    def predict(self,x):
        previousOutput = to_tensor(x)

        for currentLayer in self.Layers:
            output = tf.tensordot(previousOutput, currentLayer, axes=1)
            output = tf.sigmoid(output)
            previousOutput = output

        return output.numpy()

    def plotPredictions(self,axes=[-1.5,1.5,-1.5,1.5],lr=1,epochs=1000):
        x = np.linspace(axes[0],axes[1],100)

        y = np.linspace(axes[2],axes[3],100)

        x,y = np.meshgrid(x,y)

        X = np.c_[x.ravel(),y.ravel()]


        fig, ax = plt.subplots()
        x1 = extract(self.x,0)
        x2 = extract(self.x,1)

        def animate(i):
            ax.clear()
            self.train(lr,epochs=1)
            y_pred = self.predict(X).reshape(x.shape)
            plt.title("epoch:"+str(i))
            ax.scatter(x1,x2,alpha=1,c=self.y,s=3)
            ax.contourf(x, y, y_pred, cmap=plt.cm.brg, alpha=0.2)

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



Layers = [6,4]

A = NN(x[0],x[1],Layers,
       activations="tanh",
       initializations="xavier_uniform",
       loss_function="BCE",
       batch_size=30)
print(A)
A.plotPredictions(lr=3,epochs=1000)

# B = NN(xXOR,yXOR,[4],activations="sigmoid",initializations="xavier_uniform",)
# B.plotPredictions()






