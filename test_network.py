import numpy as np
import pandas as pd
import math as m

#IN ORDER TO RUN TRAINING, you'll need to download the .csv version of the MNIST dataset from 
#https://www.kaggle.com/datasets/oddrationale/mnist-in-csv and place mnist_train.csv and mnist_test.csv
#in the mnist-dataset directory.

class Neural_Network:
    
    def __init__(self, epochs, learningRate, batchSize):
        
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize

        #Reads in MNIST training set
        self.data_initial = pd.read_csv('./mnist-dataset/mnist_train.csv')
        self.labels = (self.data_initial['label']).to_numpy().reshape(60000,1) #(60000 imgs,)
        self.data = (self.data_initial.drop('label', axis=1)).to_numpy().reshape(60000,784) #(60000 imgs, 784, pixels)
        self.trainingSetSize = self.labels.size
        
        #Reads in MNIST test set
        self.testInitial = pd.read_csv('./mnist-dataset/mnist_test.csv')
        self.testLabels = (self.testInitial['label']).to_numpy().reshape(10000,1) #(10000 imgs,1)
        self.testData = (self.testInitial.drop('label', axis=1)).to_numpy().reshape(10000,784) #(10000 imgs, 784, pixels)
        self.testSetSize = self.testLabels.size

        self.a_0 = np.empty([784, 1]) 

        self.W_1 = np.random.default_rng().normal(0, 1, size=(16,784)) #Randomly intialized weight matrix
        self.b_1 = np.random.default_rng().normal(0, 1, size=(16,1)) #Randomly intialized bias column vector
        self.z_1 = np.empty([16, 1])
        self.a_1 = np.empty([16, 1])
        self.error_1 = np.empty([16, 1])
        
        self.W_2 = np.random.default_rng().normal(0, 1, size=(16,16))
        self.b_2 = np.random.default_rng().normal(0, 1, size=(16,1))
        self.z_2 = np.empty([16, 1])
        self.a_2 = np.empty([16, 1])
        self.error_2 = np.empty([16, 1])
       
        self.W_3 = np.random.default_rng().normal(0, 1, size=(10,16))
        self.b_3 = np.random.default_rng().normal(0, 1, size=(10,1))
        self.z_3 = np.empty([10, 1])
        self.a_3 = np.empty([10, 1])
        self.error_out = np.empty([10, 1])

        self.dW_1 = np.empty([16, 784])
        self.dB_1 = np.empty([16,1])
        self.dW_2 = np.empty([16,16])
        self.dB_2 = np.empty([16,1])
        self.dW_3 = np.empty([10,16])
        self.dB_3 = np.empty([10,1])
        
        self.y = np.zeros([1,10]) 

    def feedForward(self, x, dataset):
        #Calculates all the activations in the network for the training example, x.

        #Grabs image pixel information from the xth row of the dataset. 
        #This gives us a numpy (784,1) colm vector of activations for a training example, x, 
        #on Layer 0 (input layer)

        #1 means feedForward a training example from the training dataset
        if dataset == 1:
            self.a_0 = self.data[0, :].reshape(784,1)

        #2 means feedForward a training example from the test dataset.
        if dataset == 2:
            self.a_0 = self.testData[0, :].reshape(784,1)

        #Going into Layer 1
        self.z_1 = (np.dot(self.W_1, self.a_0)) + self.b_1
        self.a_1 = self.sigmoid(self.z_1)

        #Going into Layer 2
        self.z_2 = (np.dot(self.W_2, self.a_1)) + self.b_2
        self.a_2 = self.sigmoid(self.z_2)

        #Going into Layer 3 (output layer)
        self.z_3 = (np.dot(self.W_3, self.a_2)) + self.b_3
        self.a_3 = self.sigmoid(self.z_3)
        
#    
    def backProp(self, x): 
        #Calculates the "error" on all the neurons in the network for a training example, x.
        
        #Creates the y column vector that represents the ideal output for all the output neurons for the spesific training example.
        label = self.labels[x, 0]
        self.y[0,label] = label
        self.y = np.transpose(self.y)


        #********************************************************************************************
        #Calculuate the error on the output neurons
        self.error_out = (self.a_3 - self.y) * self.dSigmoid(self.z_3)

        #Calculate the error on each of neurons on each of the layers. Calculating backwards.
        self.error_2 = np.dot((np.transpose(self.W_3)), self.error_out) * self.dSigmoid(self.z_2)
        self.error_1 = np.dot((np.transpose(self.W_2)), self.error_2) * self.dSigmoid(self.z_1)

        self.y = np.zeros([1,10])

    def accumulateGradients(self):
        #Calculates the derivative of the cost function WRT all the weights and biases. <-- Gradient information
        #"Accumulating" (i.e. adding together) the graidents (element wise) of each of the training examples that go through.

        self.dW_1 += (np.dot(self.error_1, np.transpose(self.a_0))) #(16,1) dot (1,784) = (16,784)
        self.dB_1 += self.error_1 #(16,1)
        
        self.dW_2 += (np.dot(self.error_2, np.transpose(self.a_1))) #(16,1) dot (1,16) = (16,16)
        self.dB_2 += self.error_2 #(16,1)

        self.dW_3 += (np.dot(self.error_out, np.transpose(self.a_2))) #(10,1) dot (1,16) = (10,16)
        self.dB_3 += self.error_out #(10,1)

    def applyAvgGradient(self):
        n = self.learningRate
        m = self.batchSize
        
        self.W_1 -= ((n/m)*self.dW_1)
        self.b_1 -= ((n/m)*self.dB_1)

        self.W_2 -= ((n/m)*self.dW_2)
        self.b_2 -= ((n/m)*self.dB_2)

        self.W_3 -= ((n/m)*self.dW_3)
        self.b_3 -= ((n/m)*self.dB_3)

    def startTraining(self):
        for epochs in range(self.epochs):
            print("So far finished ", epochs, " epoch")

            #60,000 examples (x), I want 1,000 examples per batch = 60 batch
            for batch in range(int(self.trainingSetSize/self.batchSize)):
                for x in range(self.batchSize): #1,000 per batch
                    self.feedForward(x, 1)
                    self.backProp(x)
                    self.accumulateGradients()
                self.applyAvgGradient()
                
    def evaluate(self):
        #Evaluates the models accuracy by running through the 10,000 test examples and seeing how many test examples the model gets right.
        
        correct = 0
        for x in range(self.testSetSize):
            
            testLabel = self.testLabels[x, 0] #Must return a scalar of the activation
            
            self.feedForward(x, 2) #Finds new a_3 for a 
            
            #Determine which activation is biggest in final layer. I.e. which neuron is the model's choice?
            a_3 = (self.a_3).reshape(1,10).flatten() # (1,10)
            
            biggestActivation = 0
            theIndexOfBiggest = 0
            for i in range(a_3.size):
                if (a_3[i] >= biggestActivation):
                    biggestActivation = a_3[i]
                    theIndexOfBiggest = i

            if (theIndexOfBiggest == testLabel):
                print("testLabel: ",testLabel," vs indexOfBiggest",theIndexOfBiggest)
                
                correct = correct + 1
                
        return (correct/self.testSetSize)*100
        
    def sigmoid(self, colVector):
        return 1/(1 + m.e**(-1*colVector))

    def dSigmoid(self, colVector):
        return (self.sigmoid(colVector)) * (1 - self.sigmoid(colVector))

        

#12960 weights, 42 biases
nn = Neural_Network(1, 0.1, 1000)
nn.startTraining()
print("Accuracy of Model:", nn.evaluate(), "%")

#I've noticed that whether after 5 epochs, 1 epoch, 0.1 LR or 1 LR, epoch, each time with random weights and biases, we keep on "converging" onto a single value of 9.8%
#Interestingly, after testing

#I'm gonna test for overfitting.