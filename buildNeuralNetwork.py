import tensorflow as tf
import numpy as np



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
    def __init__(self,data,targets,layers):
        self.x = to_tensor(data)
        self.y = to_tensor(targets)
        self.Layers = []
        self.numberOfLayers = len(layers)
        self.layerNeurons = layers
        nextLayer = targets.shape[1]


        for index,layer in enumerate(layers[::-1]):

            if index==0:
                newLayer = tf.random.uniform(shape=[layer,nextLayer],minval=0.,maxval=1.,dtype=tf.float32)
                self.Layers.append(newLayer)
                nextLayer = layer
                continue

            newLayer = tf.random.uniform(shape=[layer,nextLayer],minval=0.,maxval=1.,dtype=tf.float32)
            self.Layers.append(newLayer)
            nextLayer = layer

        newLayer = tf.random.uniform(shape=[data.shape[1],nextLayer],minval=0.,maxval=1.,dtype=tf.float32)
        self.Layers.append(newLayer)
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

    def train(self,lr=1):
        for i in range(1500):
            with tf.GradientTape(persistent=True) as t:
                for weights in self.Layers:
                    t.watch(weights)

                previousOutput = self.x

                for currentLayer in self.Layers:
                    output = tf.tensordot(previousOutput,currentLayer,axes=1)
                    output = tf.sigmoid(output)
                    previousOutput = output



                error = tf.reduce_sum(tf.square(self.y-output))




                for index,weights in enumerate(self.Layers):
                    dError_dWeights = t.gradient(error,weights)
                    self.Layers[index] -= lr*dError_dWeights

            del t

    def predict(self,x):
        previousOutput = to_tensor(x)

        for currentLayer in self.Layers:
            output = tf.tensordot(previousOutput, currentLayer, axes=1)
            output = tf.sigmoid(output)
            previousOutput = output

        return output.numpy()


x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
print(x.shape)
y = np.array([[0],[1],[1],[0]])
print(y.shape)
A = NN(x,y,[4])

print(A)
A.train()
print(A)
print(A.predict(x[0]))



