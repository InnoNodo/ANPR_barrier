import torch
import cv2
import easyocr
import numpy as np
import re

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_allowed_plates(file_path: str) -> set:
    with open(file_path, encoding='utf-8') as f:
        return set(line.strip().upper() for line in f if line.strip())

allowed_plates = load_allowed_plates('allowed_plates.txt')

model = torch.hub.load('yolov5', 'custom', path='weights/plate.pt', source='local')
model.conf = 0.4  # Confidence threshold

# EasyOCR
reader = easyocr.Reader(['ru'], gpu=False)

# Camera
cap = cv2.VideoCapture(0)

def clean_plate_text(text: str) -> str:
    return ''.join(re.findall(r'[АВЕКМНОРСТУХ0-9]', text.upper()))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Number Plate Detection
    results = model(frame)

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        plate_img = frame[y1:y2, x1:x2]

        # OCR
        ocr_result = reader.readtext(plate_img, allowlist='АВЕКМНОРСТУХ0123456789')

        if ocr_result:
            raw_text = ocr_result[0][1]
            clean_text = clean_plate_text(raw_text)

            if clean_text:
                print(f"[OCR] Распознан номер: {clean_text}")

                if clean_text in allowed_plates:
                    print("✅ Доступ разрешён — открыть шлагбаум")
                    color = (0, 255, 0)
                else:
                    print("❌ Доступ запрещён")
                    color = (0, 0, 255)
                
                # Отрисовка
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, clean_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("ANPR", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
