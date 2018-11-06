
# Simple Vanilla MLP network structure, although it does use ReLu neurons and can use Sigmoid neurons as well
# Comes with example usage, labeled parameters to manipulate, and feed forward/back propagation functions


import numpy as np
import sys


def main(argv=None):
    if argv is None:
        argv = sys.argv

    ##########################################
    # Training Data
    inputData1 = np.array([[0, 0, 1]])
    inputData2 = np.array([[0, 1, 1]])
    inputData3 = np.array([[1, 1, 1]])
    inputData4 = np.array([[1, 1, 0]])
    inputData5 = np.array([[1, 0, 0]])
    inputData6 = np.array([[0, 0, 0]])

    inputData = [inputData1, inputData2, inputData3, inputData4, inputData5, inputData6]

    outputData1 = np.array([[0, 0]])
    outputData2 = np.array([[0, 0]])
    outputData3 = np.array([[1, 1]])
    outputData4 = np.array([[1, 1]])
    outputData5 = np.array([[1, 1]])
    outputData6 = np.array([[0, 0]])

    outputData = [outputData1, outputData2, outputData3, outputData4, outputData5, outputData6]
    ##########################################
    # Testing data
    newInputRow1 = np.array([[0, 1, 0]])
    newInputRow2 = np.array([[1, 0, 1]])

    inputTestData = [newInputRow1, newInputRow2]

    newOutputRow1 = np.array([[0, 0]])
    newOutputRow2 = np.array([[1, 1]])

    ##########################################

    # Parameters to manipulate
    # InputSize, OutputSize, NumOfHidden, HiddenHeight, bias, learning rate
    InputSize = 3
    OutputSize = 2
    NumOfHidden = 0
    # HiddenHeight = 3
    HiddenHeightRange = (2, 3)
    bias = -0.1
    learningRate = 0.01



    ##########################################
    # InputSize, OutputSize, NumOfHidden, HiddenHeightRange
    net = individual(InputSize, OutputSize, NumOfHidden, HiddenHeightRange)

    # Testing simple back prop
    for x in range(1, 1000):
        for idx, row in enumerate(inputData):
            net = MLPBackProp(net, inputData[idx], outputData[idx], learningRate, bias)

    for inputs in inputTestData:
        print("For input: ", inputs)
        finalGuess = MLPFeedForward(net, inputs, bias)
        print("Guess is ", finalGuess[len(finalGuess) - 1])






# Feed forward calculations given an MLP, returns the layers
def MLPFeedForward(givenIndividual, inputDataRow, bias):
    layersList = []

    layer1 = reLu(np.dot(inputDataRow, givenIndividual[0])+bias)
    layersList.append(layer1)

    lastHidden = layer1

    for idx, matrices in enumerate(givenIndividual):

        if idx < len(givenIndividual) - 1 and idx > 0:

            if idx == 1:
                hiddenLayer = reLu(np.dot(layer1, givenIndividual[idx])+bias)
                lastHidden = hiddenLayer
                layersList.append(hiddenLayer)

            if idx > 1:
                hiddenLayer = reLu(np.dot(hiddenLayer, givenIndividual[idx])+bias)
                lastHidden = hiddenLayer
                layersList.append(hiddenLayer)

    if len(givenIndividual) > 1:
        lastLayer = reLu(np.dot(lastHidden, givenIndividual[len(givenIndividual) - 1])+bias)
        layersList.append(lastLayer)

    return layersList


# Back propagate a single network once
# Parent is array of weight matricies, inputRowData is a single input layer, outputRowData is a single output row
# and learning rate is a float
def MLPBackProp(parent, inputRowData, outputRowData, learningRate, bias):
    # Assign child to parent
    child = parent

    # Keep track of deltas
    deltaList = []

    # Calculate layers/activations of each layer
    layersList = MLPFeedForward(child, inputRowData, bias)

    # Iterate through layers and calculate errors and deltas of each
    for idx, lay in enumerate(layersList):

        # Iterate backwards through the layer list
        if (len(layersList) - 1) - idx > 0:
            outer = layersList[(len(layersList) - 1) - idx]

            # Check if output layer, delta and error are the same thing
            if idx == 0:
                previousWeight = 'None, outerlayer'
                error = outputRowData - layersList[len(layersList) - 1]
                delta = error

            # When iterating backwards through the layersList, check for index out of bounds exception
            # Calculates errors and deltas for hidden layers, not last or first layers
            if len(layersList) - 1 - idx - 1 >= 0 and idx != 0:
                error = delta.dot(previousWeight.T)
                delta = error * reLuDerivative(outer)

            previousWeight = child[(len(child) - 1) - idx]

        # Last one
        if (len(layersList) - 1) - idx == 0:

            # Case where net has only one layer
            if idx == 0:
                outer = layersList[0]
                previousWeight = 'None, outerlayer'
                error = outputRowData - layersList[len(layersList) - 1]

            # Net with more than one layer, is last delta/error calculation
            if idx != 0:
                outer = layersList[0]
                previousWeight = child[1]
                error = delta.dot(previousWeight.T)

            delta = error * reLuDerivative(outer)

        # Append delta to list
        deltaList.append(delta)

    # Adjust weights
    for idx, weights in enumerate(child):

        # If idx is not on last weight matrix, add
        if idx != (len(child) - 1):
            child[len(child) - 1 - idx] += learningRate * (layersList[len(layersList) - 2 - idx].T.dot(deltaList[idx]))

        if idx == (len(child) - 1):
            child[len(child) - 1 - idx] += learningRate * (inputRowData.T.dot(deltaList[idx]))

    return child


# Individual basic neural net.
# This takes in Input vector size, Output vector size, the number of hidden layers, and a range for the heights of the
# hidden layers.
# Returns a list of weight matrices.
def individual(InputSize, OutputSize, NumOfHidden, HiddenHeightRange):
    WeightMatricies = []
    HiddenHeights = []

    # If there is no hidden layers, just create weight matrix between input and output. Disregard HiddenHeight
    if NumOfHidden == 0:
        # Initial matrix from input to output layer
        inputWeightMatrix = 2 * np.random.random((InputSize, OutputSize)) - 1
        WeightMatricies.append(inputWeightMatrix)

    if NumOfHidden >= 1:

        # Create list of hidden layer heights
        for x in range(NumOfHidden):
            tempHeight = np.random.random_integers(HiddenHeightRange[0], HiddenHeightRange[1])
            HiddenHeights.append(tempHeight)

        # Initial matrix from input to first hidden layer
        inputWeightMatrix = 2 * np.random.random((InputSize, HiddenHeights[0])) - 1
        WeightMatricies.append(inputWeightMatrix)

        # If more than one hidden layer, and thus more than two matrices
        if NumOfHidden > 1:

            # Iterate through hidden layers
            for x in range(NumOfHidden):

                # Since there is more than one hidden layer, start by creating matrix between the first
                # two hidden layers
                if x == 0:
                    M2 = 2 * np.random.random((HiddenHeights[0], HiddenHeights[1])) - 1
                    WeightMatricies.append(M2)

                # If the hidden layer is not the initial nor the last, create a matrix between the two
                if x != 0 and x != NumOfHidden - 1:
                    MN_1 = 2 * np.random.random((HiddenHeights[x], HiddenHeights[x+1])) - 1
                    WeightMatricies.append(MN_1)

        # Add matrix between output and last hidden layer
        outputWeightMatrix = 2 * np.random.random((HiddenHeights[len(HiddenHeights) - 1], OutputSize)) - 1
        WeightMatricies.append(outputWeightMatrix)

    # Returns the random weight matrices for each transition between hidden layers
    return WeightMatricies





# ReLu function
def reLu(x):
    return np.maximum(0, x)


def reLuDerivative(x):
    y = (x > 0) * 1
    return y
    # if x > 0:
    #     return 1
    # if x <= 0:
    #     return 0


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
    return x * (1 - x)


if __name__ == '__main__':
    main()
