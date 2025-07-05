import cv2
import easyocr
from gate import open_gate
from camera import get_frame
import time

reader = easyocr.Reader(['en'], gpu=False)

# Загружаем список разрешённых номеров
with open("allowed_plates.txt", "r") as f:
    allowed_plates = set(line.strip().upper() for line in f.readlines())

def recognize_plate(frame):
    results = reader.readtext(frame)
    for bbox, text, conf in results:
        clean_text = text.upper().replace(" ", "")
        if 5 <= len(clean_text) <= 10:
            return clean_text
    return None

while True:
    frame = get_frame()

    if frame is None:
        continue

    # Отобразим превью
    cv2.imshow("Camera", frame)

    plate = recognize_plate(frame)
    if plate:
        print(f"[INFO] Распознан номер: {plate}")
        if plate in allowed_plates:
            print("[ACCESS] Доступ разрешён.")
            open_gate()
        else:
            print("[ACCESS] Доступ запрещён.")

        time.sleep(5)  # чтобы не распознавать один и тот же номер снова

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
