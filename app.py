from flask import Flask, render_template, Response
import cv2
import paho.mqtt.publish as publish
import time

app = Flask(__name__)

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)
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

            fgmask = fgbg.apply(frame)
            contornosimg = fgmask.copy()
            contornos, _ = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            personas_detectadas = False  # Variable para rastrear si se detectaron personas

            for c in contornos:
                if cv2.contourArea(c) < 500:
                    continue

                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                personas_detectadas = True

                # Si no se detectaron personas, enviar un mensaje MQTT
                if not personas_detectadas:
                    publish.single(MQTT_TOPIC, "sin personas", hostname=MQTT_BROKER)

                # Verificar si han pasado 3 segundos desde la última detección
                if time.time() - last_detection_time >= 3:
                    # Envía mensaje MQTT cuando se detecta una persona
                    publish.single(MQTT_TOPIC, "personas", hostname=MQTT_BROKER)
                    last_detection_time = time.time()

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
