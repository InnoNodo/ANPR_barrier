import cv2

cap = cv2.VideoCapture(0)  # 0 — встроенная камера, можно заменить на IP-поток

def get_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
