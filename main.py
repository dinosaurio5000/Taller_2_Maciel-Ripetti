##############################################################################################################

# primer 1: buscar imagenes del objeto a detectar
# para eso se puede usar google o fotos obtenidas por nosotros mismos en formato jpg

##############################################################################################################

# version de python: 3.10.5 

# descargar y instalar las librerias necesarias
# pip install onnxruntime==1.15.0
# pip install numpy==1.24.4
# pip install labelme==5.4.1
# pip install labelme2yolo==0.1.4
# pip install ultralytics==8.3.15

##############################################################################################################

# paso 2: abrir labelme
# configurar guardado automático y cambiar dirección de salida (verificar que estamos en la misma carpeta)
# crear poligono y segmentar el objeto. Luego guardar y nombrar la etiqueta
# repetir el proceso para cada imagen
# en la carpeta deberíamos tener las imagenes y las etiquetas en formato json

##############################################################################################################

# paso 3: convertir a YOLO
# en consola colocar lo siguiente:
# labelme2yolo --json_dir "direccion donde estan las etiquetas"
# se genera una carpeta llamada YOLODataset
# IMPORTANTE
# todas las imaganes de las carpetas se copian en la carpeta images y se eliminan las carpetas
# lo mismo se hace con label
# se cambia el nombre de la carpeta YOLODataset por train
# esta carpeta se copia y se crea una carpeta llamada val (lo ideal es crear muchas imagenes para el dataset y dejar algunas aca todas diferentes)
# de la carpeta val, se elimna el archivo dataset.yaml
# el archivo dataset.yaml se copia de la carpeta train se mueve a la carpeta superior
# crear una nueva carpeta en la carpeta superior llamada data donde se mueve las carpetas train y val
# el archivo dataset.yaml se copia y se pega en la carpeta superior

##############################################################################################################

# paso 4: modificar el archivo dataset.yaml
# modificar las direcciones de train y val
# eliminar la dirección de test

##############################################################################################################

# paso 5: entrenar el modelo
# en consola colocar lo siguiente:
# yolo task=segment mode=train epochs=30 data=dataset.yaml model=yolov8m-seg.pt imgsz=640 batch=2
# donde queda el modelo entrenado?
# en la carpeta runs/segment/train/weights/best.pt
# tomamos ese modelo y lo dejamos al nivel del codigo

##############################################################################################################

# paso 6: chequeo en tiempo real
# importacion de librerias
from ultralytics import YOLO
import cv2

# cargar el modelo
model = YOLO("best_s.pt")

# realizar apertura de la camara
cap = cv2.VideoCapture(1)

# realizar un bucle para capturar imagenes
while True:
    # lectura de los fotogramas
    ret, frame = cap.read()
    
    # predicción de la imagen y obtenemos los segmentos con predicción sobre el 90%
    results = model.predict(frame, imgsz=640, conf=0.90)
    
    # mostramos los resultados
    predicciones = results[0].plot()
    
    # mostramos nuestros fotogramas
    cv2.imshow("Prueba Segmentación", predicciones)

    # cerrar el programa (cuando se presione la tecla esc)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()









