#Добавление библиотек 

from ultralytics import YOLOv10 #Модель нейросети последняя версия
import cv2
import numpy as np


#Загрузка модели 
model = YOLOv10()

# Список цветов для различных классов
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]


#Трекинг в real-time 
capture = cv2.VideoCapture(0) 

#Для трекинга по готовой видеозаписи
#capture = cv2.VideoCapture(video_path) 

#=====Получение параметров видео========

#Кадры в секунду
fps = int(capture.get(cv2.CAP_PROP_FPS))

#Размерность
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True: 
    #Захват кадра 
    ret, frame = capture.read()
    if not ret: 
        break 

    # Обработка кадра при помощи модели 
    results = model(frame)[0]

    # Данные об объектах
    classes_names = results.names
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
    # Рисование рамок и данных 
    for class_id, box in zip(results.boxes.cls.cpu().numpy(),
                             results.boxes.xyxy.cpu().numpy().astype(np.int32)):
        class_name = results.names[int(class_id)]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)

    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()