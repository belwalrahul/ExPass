import os
import cv2
import time
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "C:/Study/612/Project"
face_cascade = cv2.CascadeClassifier(path + '/haarcascade_frontalface_default.xml')
# names = ["", "Rahul Belwal", "Tushar Sharma", "SDJ", "Belu", "Tanmay", "Gargi"]
names = open(path + "/namesdB.txt", "r").read().split(",")
print(names)


def readdB():
    dirlist = os.listdir(path + '/facedB/')

    images = []
    labels = []
    for item in dirlist:
        img = cv2.imread(path + '/facedB/'+ item, 0)
        images.append(img)
        labels.append(int(item.split('_')[0]))

    return images, labels

def capPics():
    _, labels = readdB()
    max_label = 0 if (len(labels) == 0) else max(labels)
    cntr1 = 0

    name = input('Your Name ---> ')
    exist = False
    list_dB = os.listdir(path + '/facedB/')

    for image in list_dB:
        userName = image.split("_")[1]
        if name == userName:
            print(userName + " you already there")
            max_label = names.index(name) - 1
            cntr1 = cntr1 if ( cntr1 > int(image.split("_")[2][:-4])) else int(image.split("_")[2][:-4])
            exist = True
            print(image)
        print(userName)

    if not exist:
        names.append(name)
        open(path + "/namesdB.txt", "a").write("," + name)


    print(len(labels))
    # vid = cv2.VideoCapture("C:/Users/belwa/Downloads/Hellrazr.mp4")
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        cv2.imshow('camera', frame)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            vid.release()
            welcome()
            break
        elif k == ord('c'): 
            cntr1 = cntr1 + 1
            cv2.imwrite(path + '/facedB/'+ str(max_label+1) +'_'+ name +'_'+ str(cntr1) +'.jpg', gray[y:y+h, x:x+w])


def training():
    images, labels = readdB()

    #test
    for i in labels:
        print(i)
        
    recognizer.train(images, np.array(labels))


def prediction():

    prev_frame_time = 0
    new_frame_time = 0
    vid = cv2.VideoCapture(0)
    while True:
        guessed_label = str(0)
        #frame = cv2.imread("C:/users/rahul/desktop/opencv-face-recognition-python-master/test-data/test6.jpg")
        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        _, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img = gray

        (xx, yy, ww, hh) = (0, 0, 0, 0)
        faces = face_cascade.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            (xx, yy, ww, hh) = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            img = gray[y:y+h, x:x+w]

        actual_label = '1'
        guessed_label, confidence = recognizer.predict(img)
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time

        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_PLAIN, 1.7, (100, 255, 0), 3, cv2.LINE_AA)

        print(names[guessed_label])
        print(confidence)
        """
        if guessed_label == actual_label:
            print('Face is Correctly Recognized as '+ gussed_label +'with confidence'+ confidence +'.')
        else:
            print('Face is not Recognized!!!!')
        """
        cv2.putText(frame, names[guessed_label], (xx, yy-5), cv2.FONT_HERSHEY_PLAIN, 1.7, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            vid.release()
            break

    welcome()
        

def recog():
    training()
    prediction()

def welcome():
    print('Menu : ')
    print('  1. Capture Photos')
    print('  2. Run Recognizer')
    ch = int(input('\nYour choice ---> '))

    if ch == 1:
        capPics()
    elif ch == 2:
        recog()
    else:
        print('Wrong Choice!!!!')
        

welcome()
