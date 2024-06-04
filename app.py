from flask import Flask, render_template, Response
import cv2
import paho.mqtt.publish as publish
import time
import numpy as np
import imutils
from Adafruit_IO import Client

app = Flask(__name__)

# Credenciales para Adafruit IO dash
ADAFRUIT_IO_KEY = ''
ADAFRUIT_IO_USERNAME = ''
aio = Client(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cv2.ocl.setUseOpenCL(False)

paused = False

MQTT_BROKER = "192.168.1.189"
MQTT_TOPIC = "demo"


contador_movimiento = 0

def send_to_adafruit_io(contador):
    aio.send('movement-events', contador)
def detect_people():
    global contador_movimiento
    last_detection_time = time.time()
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Redimensionar el frame utilizando imutils
            frame = imutils.resize(frame, width=640)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Dibujar un rectángulo en frame para señalar el estado
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
            color = (0, 255, 0)
            texto_estado = "Estado: No se ha detectado movimiento"

            # Especificar los puntos extremos del área a analizar
            area_pts = np.array([[0, 230], [100, 200], [490, frame.shape[0]], [0, frame.shape[0]]])

            # Con ayuda de una imagen auxiliar, determinar el área sobre la cual actuará el detector de movimiento
            imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
            imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
            image_area = cv2.bitwise_and(gray, gray, mask=imAux)

            # Obtener la imagen binaria donde la región en blanco representa la existencia de movimiento
            fgmask = fgbg.apply(image_area)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.dilate(fgmask, None, iterations=2)

            # Encontrar los contornos presentes en fgmask, para luego basándonos en su área determinar si existe movimiento
            cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            personas_detectadas = False
            for cnt in cnts:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    texto_estado = "Estado: Alerta Movimiento Detectado!"
                    color = (0, 0, 255)
                    personas_detectadas = True

            # Enviar mensaje MQTT basado en detección de personas
            if personas_detectadas:
                if time.time() - last_detection_time >= 3:
                    publish.single(MQTT_TOPIC, "personas", hostname=MQTT_BROKER)
                    last_detection_time = time.time()
            else:
                publish.single(MQTT_TOPIC, "sin personas", hostname=MQTT_BROKER)

            # Envía el contador de movimiento a Adafruit IO
            if contador_movimiento > 0:
                send_to_adafruit_io(contador_movimiento)

            # Visualizar el área que vamos a analizar y el estado de la detección de movimiento
            if isinstance(frame, np.ndarray):
                cv2.drawContours(frame, [area_pts], -1, color, 2)
                cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            yield (b'--frame\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_people(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control_flujo')
def control_flujo():
    return "Control de flujo recibido"

if _name_ == '_main_':
    app.run(host='0.0.0.0',debug=True)