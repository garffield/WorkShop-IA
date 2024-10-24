import cv2
import time
import numpy as np

# Definindo uma variável que ira armazenar as cores da caixa a ser utilizadas durante a execução
colors = [(0, 255, 255), (255, 255, 0), (0,255, 0), (255, 0, 0)]

# Usando o open, lendo as linhas, retirando espaçoes em branco e inserindo na lista
class_names = []

# Abrir o arquivo "coco.names" e ler todas as informações linha por linha
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Funcao "VideoCapture" da biblioteca "cv2" sendo atribuida à variável "cap"
cap = cv2.VideoCapture(0) #0 

# Lê a rede de aprendizado profundo representada em um dos formatos suportados
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Carregar modelo de detecção
model = cv2.dnn_DetectionModel(net)

# Carregar o padrão de entrada (resolução e escala) utilizada no modelo
model.setInputParams(size =(416, 416), scale = 1/255)

# Classes que ele é (Pessoa)
# Score - Confiança, o quão preciso é essa detecção
# Boxes - Os 4 pontos de de
while True:

    x, frame = cap.read()

    start = time.time()

    classes, scores, boxes = model.detect(frame, 0.1, 0.3)

    end = time.time()

    # Color pega o color lá de cima e efetua uma operação matemática, onde a mesma classe id terá a mesma cor
    # Label adiciona a classe id detectada ao nome (pegando pelo id do arquivo de nomes).
    # cv2.retangle (Desenhando a box no frame, a cor e a espessura)
    # cv2.puttext - Escrevendo no frame (-10 acima da box)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = colors[int(classid) % len(colors)]
        label = f"{class_names[classid]}:{score}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Escrevendo o FPS na imagem
    # Volta para o laço While
    fps_label = f"FPS: {round((1.0 / (end-start)), 2)}"
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("detections", frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()