import numpy as np 
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

###############################
path='myData'
testRatio=0.2
valRatio=0.2
batchsizeval=50
epochsval=15
stepsperepoch=130
###############################
images=[]
classNo=[]
myList=os.listdir(path)
print(myList)
no_of_classes=len(myList)

for x in range (0,no_of_classes):
	my_pic_list= os.listdir(path+"/"+str(x))
	for y in my_pic_list:
		curImg = cv2.imread(path+"/"+str(x)+"/"+y)
		curImg = cv2.resize(curImg,(32,32))
		images.append(curImg)
		classNo.append(x)
print(" ")

images = np.array(images)
classNo = np.array(classNo)

x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=testRatio)
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=valRatio)

def preprocessing(img):
		img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img=cv2.equalizeHist(img)
		img=img/255
		return img 


x_train=np.array(list(map(preprocessing,x_train)))
x_test=np.array(list(map(preprocessing,x_test)))
x_validation=np.array(list(map(preprocessing,x_validation)))

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation=x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

datagen=ImageDataGenerator(width_shift_range=0.1, 
							height_shift_range=0.1,
							zoom_range=0.2,
							shear_range=0.1,
							rotation_range=10)
datagen.fit(x_train)
y_train=to_categorical(y_train,no_of_classes)
y_test=to_categorical(y_test,no_of_classes)
y_validation=to_categorical(y_validation,no_of_classes)

def myModel():
	noOfFilters=60
	sizeOfFilter1=(5,5)
	sizeOfFilter2=(3,3)
	sizeofPool=(2,2)
	noOfNode=500

	model = Sequential()
	model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(32,32,1),activation='relu')))
	model.add((Conv2D(noOfFilters,sizeOfFilter1,activation='relu')))
	model.add(MaxPooling2D(pool_size=sizeofPool))
	model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
	model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu')))
	model.add(MaxPooling2D(pool_size=sizeofPool))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(noOfNode,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(no_of_classes,activation='softmax'))
	model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
	return model

model = myModel()

history=model.fit(datagen.flow(x_train,y_train,batch_size=batchsizeval),
							steps_per_epoch=stepsperepoch,
							epochs=epochsval,
							validation_data=(x_validation,y_validation),
							shuffle=1)

tf.keras.models.save_model(model,"trained_model.h5","wb")