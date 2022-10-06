import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

"""
Function that normalizes the train and test data by using the standard method.
The normalization is done by subtracting the mean of the attributes of the
training data and dividing the result by the standard deviation.
Part of this function was taken from lab 4.
"""
def normalizeData(trainData, testData):
    mean = np.mean(trainData, axis=0)
    std = np.std(trainData, axis=0)
    trainData = trainData-mean
    trainData = trainData/std
    testData = testData-mean
    testData = testData/std

    return trainData, testData


"""
Function that reads the train and validation data and converts them to nparrays.
"""
def readTrainAndValidationData():
    f = open("./data/train.txt")
    aux = f.readline()          # Reading the header of the file.
    trainImagesOriginal = []
    trainLabelsOriginal = []
    for line in f:
        aux = line.strip().split(",")           # Getting the label and the class of the image.
        img = io.imread("./data/train+validation/{}".format(aux[0]))            # Reading the features of the image.
        dataImage = np.asarray(img).reshape(-1)         # Converting the image to an 1D nparray, by flattening the 16x16x3 matrix.
        trainImagesOriginal.append(dataImage)
        trainLabelsOriginal.append(int(aux[1]))
    f.close()
    trainImages = np.array(trainImagesOriginal).astype('float64')           # Creating the train images nparray.
    trainLabels = np.array(trainLabelsOriginal)                             # Creating the train labels nparray.

    f = open("./data/validation.txt")
    aux = f.readline()          # Reading the header of the file.
    validationImagesOriginal = []
    validationLabelsOriginal = []
    for line in f:
        aux = line.strip().split(",")           # Getting the id and the class of the image.
        img = io.imread("./data/train+validation/{}".format(aux[0]))            # Reading the features of the image.
        dataImage = np.asarray(img).reshape(-1)         # Converting the image to an 1D nparray, by flattening the 16x16x3 matrix.
        validationImagesOriginal.append(dataImage)
        validationLabelsOriginal.append(int(aux[1]))
    f.close()
    validationImages = np.array(validationImagesOriginal).astype('float64')           # Creating the validation images nparray.
    validationLabels = np.array(validationLabelsOriginal)                             # Creating the validation labels nparray.

    return trainImages, trainLabels, validationImages, validationLabels


"""
Function that normalizes the train and validation data and predicts the labels on the validation data.
This function is used to measure the accuracy of the classifier on the validation data.
"""
def normalizeAndPredictOnValidationData(trainImages, trainLabels, validationImages, validationLabels):
    trainImages, validationImages = normalizeData(trainImages, validationImages)

    model = DecisionTreeClassifier()           # Initializing the model.
    model.fit(trainImages, trainLabels)        # Training the model.
    predictedLabels = model.predict(validationImages)       # Making predictions on the validations data.

    print("Accuracy on validation data:", accuracy_score(validationLabels, predictedLabels))
    confMatrix = confusion_matrix(validationLabels, predictedLabels, labels=model.classes_)     # Displaying the confusion matrix.
    display = ConfusionMatrixDisplay(confMatrix, display_labels=model.classes_)
    display.plot()
    plt.show()

# Getting the data.
trainImages, trainLabels, validationImages, validationLabels = readTrainAndValidationData()

normalizeAndPredictOnValidationData(trainImages, trainLabels, validationImages, validationLabels)

