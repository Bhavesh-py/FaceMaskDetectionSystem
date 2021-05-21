from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


#Setting up initial learning rate. number of epochs and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"F:\Study\FaceMaskDetection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

#At times these print statements are used to let us know the current point of execution.
print("LOADING IMAGES...")

data=[]
labels=[]

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(244,244))
        image = img_to_array(image)
        image = preprocess_input(image)
        #We are using preprocess_input because we are using mobilenet_v2net

        data.append(image)
        labels.append(category)

#Performing one-hot encoding on labels
print("PERFORMING ONE-HOT ENCODING...")
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

#Splitting data into train and test
(trainX, testX, trainY, testY) = train_test_split(data, labels , test_size=0.20, stratify=labels, random_state=42)

#Loading the MobileNetV2.
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor = Input(shape=(244,244,3)))

#Creating the fully connected layer.
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


#placing the fully connected model on the top of base model.
model = Model(inputs=baseModel.input, outputs=headModel)
 
 #Freezing all the layers in base model so that they don't get updated suring the first training process.
for layer in baseModel.layers:
    layer.trainable = False

#compiling our model
print("COMPILING MODEL...")
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])


print("TRAINING HEAD...")
H = model.fit(
    trainX, trainY, batch_size=BS,
    steps_per_epoch = len(trainX // BS),
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

#Making predictions on the testing set
print("EVALUATING NETWORK...")
predIdxs = model.predict(testX, batch_size=BS)


#Finding the label for each image in the testing set.
predIdxs = np.argmax(predIdxs, axis=1)

#Printing Classification report.
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

#Saving the trained model
print("SAVING MODEL...")
model.save("mask_detector.model", save_format="h5")

#Plotting the training loss and accuracy
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("EPOCH")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("TLA_Plot.png")
