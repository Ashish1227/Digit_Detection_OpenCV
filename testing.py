import numpy as np 
import cv2
import tensorflow as tf
#################################
width = 640
height = 480
#################################

cap=cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)


model=tf.keras.models.load_model("trained_model.h5")

def preprocessing(img):
		img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img=cv2.equalizeHist(img)
		img=img/255
		return img 

while True:
	success,imgorg =cap.read()

	img=np.asarray(imgorg)
	img=cv2.resize(img,(32,32))
	img=preprocessing(img)
	#cv2.imshow("processed image",img)
	img=img.reshape(1,32,32,1)


	predictions=model.predict(img)
	pval=np.amax(predictions)
	probVal=np.argmax(predictions,axis=1)

	if pval>0.85:
		cv2.putText(imgorg,str(probVal)+"  "+str(pval),
			(50,50),cv2.FONT_HERSHEY_COMPLEX,
			1,(0,0,255),1)

	cv2.imshow("original image",imgorg)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break