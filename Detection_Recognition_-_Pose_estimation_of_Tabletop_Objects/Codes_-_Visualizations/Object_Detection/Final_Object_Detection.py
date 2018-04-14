
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import pickle

# fix random seed for reproducibility

def load_pickle(pickle_filename='dataset'):
	file=open(pickle_filename+".pickle",'rb')
	database=pickle.load(file)
	print("Dataset loaded")
	file.close()
	# normalize inputs from 0-255 to 0.0-1.0
	X=np.array(database['X'])
	X = X / 255.0
	print(database['Y'])
	return X,database['Y']

seed = 7
np.random.seed(seed)
#Loading the dataset
X,Y=load_pickle('dataset2(object)')
X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.33, random_state=42)
print(X_train.shape)

# integer encode
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test=label_encoder.fit_transform(y_test)
#One hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Convolution2D(64, 3, 3, input_shape=(240,320,1), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(300, activation='relu', W_constraint=maxnorm(3)))
#model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# checkpoint
filepath="tabletop_mouse_angle_detection.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs,callbacks=callbacks_list, batch_size=4, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


