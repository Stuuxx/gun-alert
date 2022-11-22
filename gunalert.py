import cv2
import codecs
import imutils
from datetime import datetime
import requests
import subprocess
import numpy as np

camera = cv2.VideoCapture("pistool.mp4")
success,image = camera.read()

# Face Recognition Cascade
facecascade = cv2.CascadeClassifier("C:/Users/User/Desktop/Trabalho/xmls/haarcascade_frontalface_alt.xml")

# Handgun Cascade
gun_cascade = cv2.CascadeClassifier("C:/Users/User/Desktop/Trabalho/xmls/cascadeold.xml")
firstFrame = None
gun_exist = False
count = 0

# Telegram Bot Requests
token = "5983326960:AAHhQ1xnLTHyn6qjVz0rnX1b7H0n7pUrIeg"
chat_id = "-1001810847383"

def send_msg(text):
   url_req = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text=" + text 
   results = requests.get(url_req)
   print(results.json())

def send_image(imageFile):
        command = 'curl -s -X POST https://api.telegram.org/bot' + token + '/sendPhoto -F chat_id=' + chat_id + " -F photo=@" + imageFile
        subprocess.call(command.split(' '))
        return

date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

# Execução
while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width = 750) 
    frame = cv2.flip(frame, 1)

    # Tratamento: Tornar frames cinzas e aplicar blur
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.GaussianBlur(frameGray, (21, 21), 0)
    ImagemGaussianBlur = cv2.GaussianBlur(frameGray, (7, 7), 0)

    # Gray Scale para Face Recognition
    detect = facecascade.detectMultiScale(frameGray, 1.5, 6)

    # Face Recognition
    for (x, y, w, h) in detect:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        logs = codecs.open("C:/Users/User/Desktop/Trabalho/logs/face.txt", "a+", "UTF-8")
        logs.write("Nova Face Detectada " + date + "\n")
        print('Face Detectada: ', success)

    # Gray Scale para Handgun detect
    gun = gun_cascade.detectMultiScale(frameGray, 1.8, 10, minSize = (100, 100))
    
    # Handgun Detection
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame,(x, y),(x + w,y + h),(255, 0, 0), 2)
        roi_gray = frameGray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]    
        cv2.imwrite("C:/Users/User/Desktop/Trabalho/frames/frame%d.jpg" % count, image)      
        success,image = camera.read()
        print('Arma Detectada: ', success)
        count += 1
        if count == 15:
            send_msg("Arma Detectada! " + "\n" + date)
            send_image("frames/frame1.jpg")

    if firstFrame is None:
        firstFrame = frameGray
        continue
   
    # Inserir text na camera
    cv2.putText(frame, ("Eduardo Maragno e Guilherme Zanchin"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Binarização da imagem
    _, ImagemThrehold = cv2.threshold(
        ImagemGaussianBlur, 120, 255, cv2.THRESH_BINARY_INV
    )
    ImagemAdaptive = cv2.adaptiveThreshold(
        ImagemGaussianBlur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        10,
    )
    # Correção morfológica, aplicando método open e close
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(ImagemAdaptive, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(ImagemAdaptive, cv2.MORPH_CLOSE, kernel) 

    if ret:
        cv2.imshow("Gun Alert", frame)
        cv2.imshow("Morph Open", opening)
        cv2.imshow("Morph Close", closing)   
        key = cv2.waitKey(1)
        if key == 27:
            break

cv2.destroyAllWindows()
camera.release()
