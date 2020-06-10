import cv2

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video=cv2.VideoCapture(0)

while True:
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,255),2)
	cv2.imshow("Frame",frame)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break
video.release()
cv2.destroyAllWindows()