import cv2
import numpy as np
import imutils


# Función para redimensionar un frame a un tamaño deseado
def resize_frame(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Cambiar la fuente del video a la webcam (índice 0)
# Nombre video XdxDXDxd
cap = cv2.VideoCapture(0)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Redimensiona el frame utilizando imutils
    frame = imutils.resize(frame, width=640)  # Cambia el ancho según sea necesario

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dibujamos un rectángulo en frame, para señalar el estado
    # del área en análisis (movimiento detectado o no detectado)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"

    # Especificamos los puntos extremos del área a analizar
    # Primer array: Esquina superior izquierda
    # Segundo array: Esquina superior derecha
    # Tercer array: Esquina inferior derecha
    # Cuarto array: Esquina inferior izquierda
    # London
    # area_pts = np.array([[10, 250], [300, 300], [400, frame.shape[0]], [45, frame.shape[0]]])

    # Aeropuerco
    # area_pts = np.array([[240, 320], [480, 320], [620, frame.shape[0]], [50, frame.shape[0]]])

    area_pts = np.array([[0, 230], [100, 200], [490, frame.shape[0]], [0, frame.shape[0]]])

    # Con ayuda de una imagen auxiliar, determinamos el área
    # sobre la cual actuará el detector de movimiento
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    # Obtendremos la imagen binaria donde la región en blanco representa
    # la existencia de movimiento
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Encontramos los contornos presentes en fgmask, para luego basándonos
    # en su área poder determinar si existe movimiento
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto_estado = "Estado: Alerta Movimiento Detectado!"
            color = (0, 0, 255)

    # Visualizamos el alrededor del área que vamos a analizar
    # y el estado de la detección de movimiento
    cv2.drawContours(frame, [area_pts], -1, color, 2)
    cv2.putText(frame, texto_estado, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('fgmask', fgmask)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()