import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size,))
        self.bias = np.random.uniform(low=-0.5, high=0.5)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0

    def train(self, inputs, target):
        prediction = self.predict(inputs)
        error = target - prediction

        # Update weights and bias using the perceptron learning rule
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error

        return error



class Perceptron_Layer:
    def __init__(self, n_inputs, n_neurons, learning_rate):
        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(n_inputs,  n_neurons))
        # 1 x n_neurons matrix
        self.biases = np.ones((1, n_neurons))
        self.eta = learning_rate
        self.grounds = []
        self.potentials = []

    #  takes a single input
    def forward(self, input, target, adjustment):
        #  convert the target to a vector representation

        # inputs [1x784] dot [784 x 10] + [1x10]
        activationPotential = np.dot(input, self.weights)
        # perceptron results
        # [1 x 10] result matrix

        prediction = np.argmax(activationPotential)
        activationPotential = np.where(activationPotential > 0, 1, 0)
        #  create a truthVector t^i per equation for weight adjustment
        truthVector = np.zeros(10, dtype="int")
        truthVector[target] = 1

        self.grounds.append(truthVector.argmax())
        self.potentials.append(activationPotential.argmax())

        if target == prediction:
            self.accurateCount += 1
            if (self.inputSize > 0):
                self.accuracy = self.accurateCount / self.inputSize
        else:
            self.incorrect += 1
            error = []
            for i in range(len(truthVector)):
                error.append(truthVector[i]-activationPotential[i])

            np.array(error)

            #  for HW1 algo... single batch input... get the single array input and or access weights for tuning...
            if adjustment:
                #  threshold check
                #  contains the binary classification from potential fire
                self.weights = self.weights + self.eta * np.outer(error, input).T


                #  update weights

        self.inputSize += 1

    def determineAccuracy(self):
        if self.inputSize == 0:
            print("No input data. Accuracy set to 0.")
            self.accuracy = 0.0
        else:
            self.accuracy = round((self.accurateCount / self.inputSize) * 100, 2)
            print("Accuracy: " + str(self.accuracy) + "%")

    def calculateConfusionMatrix(self):
        cm = confusion_matrix(self.grounds, self.potentials)
        return cm

    def displayConfusionMatrix(self, cm):
        print("Confusion Matrix:")
        print(cm)

    def plotConfusionMatrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=np.arange(10), yticklabels=np.arange(10))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    def clearAccuracy(self):
        self.accuracy = 0.0
        self.accurateCount = 0
        self.incorrect = 0
        self.inputSize = 0
        self.grounds = []
        self.potentials = []

    def plotAccuracyOverEpochs(self, accuracies):
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(1, len(accuracies) + 1), accuracies, marker='o')
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.show()

def getTrainingLabels(dataFrame):
    groundTruthLables = dataFrame['label'].to_numpy()
    # drop the labels of the data and then divide each array
    return groundTruthLables

def getTrainingInputs(dataFrame):
    inputs = dataFrame.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedInputs = inputs / 255
    return normalizedInputs

def shuffleTrainData():
    mnist_data = pd.read_csv("mnist_train.csv")
    # shuffle the contents of the dataset for training
    #  the prevents the perceptrons from being trained on ordering of the input data.
    shuffledMnistData = mnist_data.sample(frac=1)
    return  shuffledMnistData

def getTesingLabels(dataFrame):
    groundTruthTestLables = dataFrame['label'].to_numpy()
    # drop the labels of the data and then divide each array
    return groundTruthTestLables

def getTestingInputs(dataFrame):
    inputs = dataFrame.drop("label", axis=1).to_numpy()
    # divide each value vector by 255...
    normalizedTestInputs = inputs / 255
    return normalizedTestInputs

def shuffleTestData():
    mnist_data = pd.read_csv("mnist_test.csv")
    # shuffle the contents of the dataset for training
    #  the prevents the perceptrons from being trained on ordering of the input data.
    shuffledMnistTestData = mnist_data.sample(frac=1)
    return  shuffledMnistTestData

# input 785 inputs within 10 perceptrons each input is a vector of 785 ie 28x28+1 (+1 for bias)

def main():
    # read in the MNIST data:
    normalizedInputs = shuffleTrainData()
    y = getTrainingLabels(normalizedInputs)
    X = getTrainingInputs(normalizedInputs)
    # normalizedInputs = normalizedInputs.reshape((-1, 28, 28))
    # confirms 28 x 28
    # print(normalizedInputs.shape)
    # print(normalizedInputs[0])

    #  Model #1 - Learning Rate 0.1
    perceptronModel = Perceptron_Layer(784, 10, 0.1)

    #  single input iteration for the batch size of training being 1
    #  initial test run...
    print("Epoch: 0")
    print('=============================')
    #  Training Cycles:
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    accuracy_over_initial_epoch = []
    for d in range(len(normalizedInputs)):
        digit = X[d]
        truth = y[d]
        perceptronModel.forward(digit, truth, False)
    accuracy_over_initial_epoch.append(perceptronModel.accuracy)
    confusion_matrix_result = perceptronModel.calculateConfusionMatrix()
    perceptronModel.displayConfusionMatrix(confusion_matrix_result)
    perceptronModel.determineAccuracy()
    # Plot confusion matrix heatmap
    perceptronModel.plotConfusionMatrix(confusion_matrix_result)

    perceptronModel.plotAccuracyOverEpochs(accuracy_over_initial_epoch)
    perceptronModel.clearAccuracy()
    #  eta is 0.1%
    epochModel1Count = 1
    # eta 0.1
    print("Training on learning rate at 0.1")
    accuracy_over_epoch_w_eta = []
    while epochModel1Count <= 70:
        normalizedInputs = shuffleTrainData()
        y = getTrainingLabels(normalizedInputs)
        X = getTrainingInputs(normalizedInputs)
        for b in range((len(normalizedInputs))):

            digit = X[b]
            truth = y[b]
            perceptronModel.forward(digit, truth, True)
            accuracy = perceptronModel.accuracy
            accuracy_over_epoch_w_eta.append(perceptronModel.accuracy)
        epochModel1Count += 1
    #  clean up for 0.1 eta
    confusion_matrix_result = perceptronModel.calculateConfusionMatrix()
    perceptronModel.displayConfusionMatrix(confusion_matrix_result)
    # Plot confusion matrix heatmap
    perceptronModel.plotConfusionMatrix(confusion_matrix_result)
    perceptronModel.determineAccuracy()
    perceptronModel.plotAccuracyOverEpochs(accuracy_over_epoch_w_eta)
    perceptronModel.clearAccuracy()



main()

