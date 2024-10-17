import cv2
import time
import numpy as np
import os
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
model = YOLO("yolo_models/yolov8m.pt")
font = cv2.FONT_HERSHEY_DUPLEX
kamera = cv2.VideoCapture("videos/video.mp4")

region1 = np.array([(400, 0), (400, 720), (500, 720), (500, 0)])
region1 = region1.reshape((-1, 1, 2))
start = 0
total_fps = 0
frame_count = 0
end = time.time()
total = set()
while True:

    ret, frame = kamera.read()

    frame = cv2.flip(frame, 1)
    if not ret:
        break

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.track(rgb_img, persist=True, classes=18)
    labels = results[0].names

    cv2.line(frame, (450, 0), (450, 720), (0, 0, 255), 10)

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        cls = results[0].boxes.cls[i]
        try:
            ids = results[0].boxes.id[i]
        except TypeError:
            ids=0
        x1, y1, x2, y2, cls, ids = int(x1), int(y1), int(x2), int(y2), int(cls), int(ids)
        name = labels[cls]
        if name != 'sheep':
            continue
        cx = int(x1 / 2 + x2 / 2)
        cy = int(y1 / 2 + y2 / 2)


        inside_region1 = cv2.pointPolygonTest(region1, (cx, cy), False)
        if inside_region1 > 0:
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            total.add(ids)

    total_str = 'TOTAL: ' + str(len(total))

    end = time.time()
    fps = 1 / (end - start)
    start = end
    total_fps += fps
    frame_count += 1
    cv2.putText(frame, f"FPS sayisi: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Toplam koyun= {total_str}", (0, 30), font, 1, (0, 0, 255), 2)

    cv2.imshow("Koyun sayma kodu", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()
average_fps = total_fps / frame_count
print(f"Ortalama FPS: {average_fps:.2f}")
with open('txt/son_koyun.txt', 'w', encoding='utf-8') as file:
    file.write(f"Son Koyun Sayısı= {total_str}")





