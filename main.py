import torch
import cv2
import easyocr
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Загрузка модели YOLOv5
model = torch.hub.load('yolov5', 'custom', path='weights/plate.pt', source='local')
model.conf = 0.4  # Confidence threshold

# EasyOCR
reader = easyocr.Reader(['en'])

# Камера
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция номеров
    results = model(frame)

    # Обработка детекций
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)

        # Вырезаем номер
        plate_img = frame[y1:y2, x1:x2]

        # Распознаём номер
        ocr_result = reader.readtext(plate_img)

        # Показываем результат
        if ocr_result:
            text = ocr_result[0][1]
            print(f"[OCR] Распознан номер: {text}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
