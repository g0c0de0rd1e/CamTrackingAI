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


