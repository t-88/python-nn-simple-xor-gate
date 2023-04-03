import numpy as np
import scipy.special
import random

class NeuralNetwork():
    def __init__(self,inputCount,hiddenCount,outputCount) -> None:
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.hiddenCount = hiddenCount

        self.w_ih = np.random.random(size=(hiddenCount,inputCount)) - 0.5
        self.w_ho = np.random.random(size=(outputCount,hiddenCount)) - 0.5



        self.b_h = np.zeros(hiddenCount)
        self.b_o = np.zeros(outputCount)

        self.lr = 1000

    def initWeights(self,w_ih,w_ho):
        self.w_ih = np.array(w_ih,ndmin=2)
        self.w_ho = np.array(w_ho,ndmin=2)

    def activation(self,arr):
        return scipy.special.expit(arr)
        

    def feedForward(self,inputs):
        inputs = np.array(inputs).T

        
        hidden = np.dot(self.w_ih,inputs) + self.b_h
        hidden = self.activation(hidden)

        outputs = np.dot(self.w_ho,hidden) + self.b_o
        outputs = self.activation(outputs)


        return outputs
    
    def backpropagate(self,inputs,targets):
        inputs = np.array(inputs,ndmin=2).T
        targets = np.array(targets,ndmin=2).T

        hidden = np.dot(self.w_ih,inputs)
        hidden =  self.activation(hidden)
        outputs = np.dot(self.w_ho,hidden)
        outputs = self.activation(outputs)

        output_errors = targets - outputs
        hidden_errors = np.dot(self.w_ho.T,output_errors) 


        self.w_ho += self.lr *  np.dot( outputs *  ( 1 - outputs ) * output_errors , hidden.T)
        self.w_ih += self.lr *  np.dot( hidden * ( 1 - hidden ) * hidden_errors  , inputs.T)
        

    def train(self,inputs,outputs):
        self.backpropagate(inputs,outputs)        



def mumbleJumple(arr):
    for i in range(0,len(arr)):
        randIndex = random.randint(0,len(arr) - 1)
        tmp = arr[randIndex]
        arr[randIndex] = arr[i]
        arr[i] = tmp


inputs = [[0,0],[0,1],[1,0],[1,1]]
outputs = [0,1,1,0]
order = [0,1,2,3]


w_ih =  [[-11.19538168 ,  5.52337987],
         [  6.27218964 ,-12.60032988],
         [ -8.04042386 , -7.92286709],
         [ 10.94131991 , -5.3970543 ],]
w_ho =  [  3.85668734,   7.86897999, -14.74648505,  -4.12533873]

#0.985

w_ih =  [[-11.72005593 ,  5.78760206],
         [  6.55027893 ,-13.15437583],
         [ -8.30945956 , -8.19826511],
         [ 11.56131606 , -5.70851416],]
w_ho =  [  4.0681363  ,  8.29756613 , -15.61805072 ,  -4.33697573]

w_ih =  [[-12.08773182 ,  5.97199551],
         [  6.74217818 ,-13.5376058 ],
         [ -8.49711703 , -8.38950822],
         [ 11.97949645 , -5.91801651],]
w_ho =  [  4.2167292 ,    8.59694573 ,  -16.2266591 ,   -4.48491858]


w_ih =  [[-13.65669222,   6.75614306],
        [  7.54485867 ,-15.14391843],
        [ -9.29257633 , -9.19451513],
        [ 13.67169482 , -6.76363728],]
w_ho =  [[  4.85519935   ,9.87642955, -18.81961461,  -5.1188273 ]]

#0.992


w_ih =  [[-22.77356079 , 11.31181497],
         [ 12.1066004  ,-24.27198743],
         [-13.86949254 ,-13.77833411],
         [ 22.83738235 ,-11.34372738],]
w_ho =  [[  8.75240271  ,17.66912172 ,-34.44771715,  -9.00798202]]
#0.99988



nn =  NeuralNetwork(2,4,1)
nn.initWeights(w_ih,w_ho)

####### train
# for epoch in range(100000):
#     mumbleJumple(order)
    
#     for index in order:
#         nn.train(inputs[index],outputs[index])
#     print(f"epoch {epoch} is done")
# for test in range(1000):
#     randIndex = random.randint(0,3)
#     print(f"input {inputs[randIndex]} expected {outputs[randIndex]} got {nn.feedForward(inputs[randIndex])}")
# print("w_ih = ",nn.w_ih)
# print("w_ho = ",nn.w_ho)


####### test
a = int(input("a: "))
b = int(input("b: "))
print(nn.feedForward([a,b]))

