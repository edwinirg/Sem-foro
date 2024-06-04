from flask import Flask, render_template, Response
import cv2
import paho.mqtt.publish as publish
import time
import numpy as np
import imutils

app = Flask(__name__)

cap = cv2.VideoCapture('https://vdo.ninja/?view=rVuGgv9')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cv2.ocl.setUseOpenCL(False)

paused = False

MQTT_BROKER = "172.26.49.244"
MQTT_TOPIC = "demo"

def detect_people():
    last_detection_time = time.time()
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Redimensiona el frame utilizando imutils
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Dibujamos un rectángulo en frame, para señalar el estado
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
            color = (0, 255, 0)
            texto_estado = "Estado: No se ha detectado movimiento"

            # Especificamos los puntos extremos del área a analizar
            area_pts = np.array([[0, 230], [100, 200], [490, frame.shape[0]], [0, frame.shape[0]]])

            imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
            imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
            image_area = cv2.bitwise_and(gray, gray, mask=imAux)

            fgmask = fgbg.apply(image_area)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.dilate(fgmask, None, iterations=2)

            cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            personas_detectadas = False  # Variable para rastrear si se detectaron personas

            for cnt in cnts:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    texto_estado = "Estado: Alerta Movimiento Detectado!"
                    color = (0, 0, 255)

                personas_detectadas = True

                # Si no se detectaron personas, enviar un mensaje MQTT
                if not personas_detectadas:
                    publish.single(MQTT_TOPIC, "sin personas", hostname=MQTT_BROKER)

                # Verificar si han pasado 3 segundos desde la última detección
                if time.time() - last_detection_time >= 3:
                    # Envía mensaje MQTT cuando se detecta una persona
                    if personas_detectadas:
                        publish.single(MQTT_TOPIC, "personas", hostname=MQTT_BROKER)
                        last_detection_time = time.time()

                cv2.drawContours(frame, [area_pts], -1, color, 2)
                cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Enviar un marco vacío cuando la cámara está pausada
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
