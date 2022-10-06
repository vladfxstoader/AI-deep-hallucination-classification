import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
"""
Function that normalizes the train and test data by using the standard method.
The normalization is done by dividing the attributes by 255, because each element
takes value in (0,255).
"""
def normalizeData(trainData, validationData, testData):
    trainData = trainData/255
    testData = testData/255
    validationData = validationData/255
    return trainData, validationData, testData


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
        dataImage = np.asarray(img)        # Converting the image to a 3D nparray (16x16x3 matrix).
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
        dataImage = np.asarray(img)        # Converting the image to a 3D nparray (16x16x3 matrix).
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
        dataImage = np.asarray(img)        # Converting the image to an 3D nparray (16x16x3 matrix).
        testImagesOriginal.append(dataImage)
        imagesName.append(aux)          # Saving the id of the image because it is used when writing the predicted labels
    f.close()                           # in the output file.
    testImages = np.array(testImagesOriginal).astype('float64')           # Creating the test images nparray.

    return testImages, imagesName


"""
Function that normalizes the train, validation and test data, builds the model and 
predicts the labels on the test data. This function also creates the output file.
"""
def normalizeAndPredict(trainImages, trainLabels, validationImages, validationLabels, testImages, imagesName):
    trainImages, validationImages, testImages = normalizeData(trainImages, validationImages, testImages)

    originalValidationLabels = validationLabels
    # We one-hot encode the values, meaning that every label is transformed into an array,
    # where the maximum value is found on the index correspoding to the class of the image.
    trainLabels = to_categorical(trainLabels)
    validationLabels = to_categorical(validationLabels)

    # The convolutional layers are chosen because they filter the inputs, and the filter
    # number is increasing with every layer, thus performing a better filtration. The padding
    # is chosen so that the images will not change their size.
    # The dropout layers are used to prevent overfitting.
    # Batch normalization layers are used to normalize the inputs going into the next series of layers.
    # The model is created by repeating this layout, and increasing the filters on the convolutional layers.
    # Then, the inputs are flattened and sent to the fully connected layer, obtained by dense layers.
    # The last layer is used to make the classification (thus it has 7 neurons because there are 7 classes),
    # and the softmax activation is used to select the neuron with the highest probability.
    model = Sequential([
        Conv2D(32,(3,3), input_shape=(16,16,3), padding='same', activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(2),
        Dropout(0.2),
        BatchNormalization(),
        Conv2D(128, 3, activation='relu',padding='same'),
        Dropout(0.2),
        BatchNormalization(),
        Flatten(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(7, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer=Adam(),
                  loss=CategoricalCrossentropy(from_logits=True),     # Chose this loss because
                  metrics=['accuracy'])                               # it is a multi-classification problem.
    modelHistory = model.fit(trainImages, trainLabels, epochs=25, validation_data=(validationImages, validationLabels))

    # Creating a plot that shows the evolution of the accuracy on training and validation data.
    trainAccuracy = modelHistory.history['accuracy']
    validationAccuracy = modelHistory.history['val_accuracy']
    epochs = range(1,26)
    plt.plot(epochs, trainAccuracy, 'g', label='Acuratetea pe antrenare')
    plt.plot(epochs, validationAccuracy, 'b', label='Acuratetea pe validare')
    plt.title('Acuratetea pe antrenare si validare')
    plt.xlabel('Epoci')
    plt.ylabel('Acuratete')
    plt.legend()
    plt.show()

    validationLoss, validationAccuracy = model.evaluate(validationImages, validationLabels, verbose = 2)
    predictedValidationLabels = model.predict(validationImages)     # Predicting the labels on validation data, in order to
    predictedValidationLabels = np.argmax(np.round(predictedValidationLabels), axis=1)      # create the confusion matrix.

    print("Accuracy on validation data: ", validationAccuracy)
    print("Loss on validation data: ", validationLoss)
    confMatrix = confusion_matrix(originalValidationLabels, predictedValidationLabels, labels=range(0,7))
    display = ConfusionMatrixDisplay(confMatrix, display_labels=range(0,7))
    display.plot()
    plt.show()

    predictedLabels = model.predict(testImages)
    predictedLabels = np.argmax(np.round(predictedLabels), axis=1)      # The predicted labels are returned as arrays,
                                                                    # and the index of the maximum value in each
                                                                    # of them is the class of the corresponding image.
    f = open("predictionsCNN.txt",'w')
    f.write("id,label\n")
    for i in range(0,len(predictedLabels)):
        f.write(imagesName[i] + ',' + str(predictedLabels[i]) + '\n')   # Writing the id and the predicted label
    f.close()                                                           # of each picture in the test data set.


# Getting the data.
trainImages, trainLabels, validationImages, validationLabels = readTrainAndValidationData()
testImages, imagesName = readTestData()

normalizeAndPredict(trainImages, trainLabels, validationImages, validationLabels, testImages, imagesName)
