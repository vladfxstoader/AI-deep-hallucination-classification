import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn import svm
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
Function that reads the test data and converts them to a nparray.
"""
def readTestData():
    f = open("./data/test.txt")
    aux = f.readline()          # Reading the header of the file.
    testImagesOriginal = []
    imagesName = []
    for line in f:
        aux = line.strip()           # Getting the id and the class of the image.
        img = io.imread("./data/test/{}".format(aux))            # Reading the features of the image.
        dataImage = np.asarray(img).reshape(-1)         # Converting the image to an 1D nparray, by flattening the 16x16x3 matrix.
        testImagesOriginal.append(dataImage)
        imagesName.append(aux)          # Saving the id of the image because it is used when writing the predicted labels
    f.close()                           # in the output file.
    testImages = np.array(testImagesOriginal).astype('float64')           # Creating the test images nparray.

    return testImages, imagesName


"""
Function that normalizes the train and validation data and predicts the labels on the validation data.
This function is used to measure the accuracy of the classifier on the validation data.
"""
def normalizeAndPredictOnValidationData(trainImages, trainLabels, validationImages, validationLabels):
    trainImages, validationImages = normalizeData(trainImages, validationImages)

    model = svm.SVC(C=3, kernel="poly")          # Initializing the model. The parameters were chosen by trial and error.
    model.fit(trainImages, trainLabels)         # Training the model.
    predictedLabels = model.predict(validationImages)       # Making predictions on the validations data.

    print("Accuracy on validation data:", accuracy_score(validationLabels, predictedLabels))
    confMatrix = confusion_matrix(validationLabels, predictedLabels, labels=model.classes_)     # Displaying the confusion matrix.
    display = ConfusionMatrixDisplay(confMatrix, display_labels=model.classes_)
    display.plot()
    plt.show()

"""
Function that normalizes the train and test data and predicts the labels on the test data.
This function also creates the output file.
"""
def normalizeAndPredictOnTestData(trainImages, trainLabels, testImages, imagesName):
    trainImages, testImages = normalizeData(trainImages, testImages)

    model = svm.SVC(C=5, kernel="rbf")          # Initializing the model. The parameters were chosen by trial and error.
    model.fit(trainImages, trainLabels)         # Training the model.
    predictedLabels = model.predict(testImages)       # Making predictions on the test data.

    f = open("predictionsSVM.txt",'w')
    f.write("id,label\n")
    for i in range(0,len(predictedLabels)):
         f.write(imagesName[i] + ',' + str(predictedLabels[i]) + '\n')      # Writing the id and the predicted label
    f.close()                                                               # of each picture in the test data set.


# Getting the data.
trainImages, trainLabels, validationImages, validationLabels = readTrainAndValidationData()
testImages, imagesName = readTestData()

# For getting the predictions on the validation data, uncomment the next line.
normalizeAndPredictOnValidationData(trainImages, trainLabels, validationImages, validationLabels)

# For getting the predictions on the test data and creating the output file, uncomment the next line.
# normalizeAndPredictOnTestData(trainImages, trainLabels, testImages, imagesName)