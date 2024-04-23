# Importación de librerías
import cv2

# Capturamos el vídeo
cap = cv2.VideoCapture(0)

# Llamada al método
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

# Deshabilitamos OpenCL, si no hacemos esto no funciona
cv2.ocl.setUseOpenCL(False)

while (1):
    # Leemos el siguiente frame
    ret, frame = cap.read()

    # Si hemos llegado al final del vídeo salimos
    if not ret:
        break

    # Aplicamos el algoritmo
    fgmask = fgbg.apply(frame)

    # Copiamos el umbral para detectar los contornos
    contornosimg = fgmask.copy()

    # Buscamos contorno en la imagen
    contornos, hierarchy = cv2.findContours(contornosimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializamos una variable para verificar si se detectó una persona
    persona_detectada = False

    # Recorremos todos los contornos encontrados
    for c in contornos:
        # Calculamos el área del contorno
        area = cv2.contourArea(c)

        # Si el área del contorno es menor que 500, lo ignoramos
        if area < 500:
            continue

        # Si encontramos un contorno que cumple con el área mínima,
        # dibujamos un rectángulo y marcamos que se detectó una persona
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        persona_detectada = True

        # Detenemos el bucle, ya que solo queremos detectar una persona
        break

    # Mostramos las capturas
    cv2.imshow('Camara', frame)
    cv2.imshow('Umbral', fgmask)
    cv2.imshow('Contornos', contornosimg)

    # Sentencias para salir, pulsa 's' y sale
    k = cv2.waitKey(30) & 0xff
    if k == ord("s"):
        break

# Liberamos la cámara y cerramos todas las ventanas
cap.release()
cv2.destroyAllWindows()
