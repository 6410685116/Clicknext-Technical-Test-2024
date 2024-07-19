from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 
from collections import defaultdict

model = YOLO("yolov8n.pt")
text = "Nattapat-Clicknext-Internship-2024"

track_history = defaultdict(lambda: [])

def draw_boxes(frame, boxes):    

    annotator = Annotator(frame)
    i = 0
    for box in boxes:
        x, y, w, h = box.xywh[0]
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf[0]
        track = track_history[i]
        track.append((float(x), float(y)))

        annotator.box_label(box=coordinator, label=class_name, color=(255,0,0))

        if len(track) > 20:
            track.pop(0)

        annotator.draw_centroid_and_tracks(track, color=(255, 0, 255), track_thickness=2)

    return annotator.result()

def detect_object(frame):

    results = model.track(source=frame, classes= 15, persist=True)

    for result in results:
        frame = draw_boxes(frame, result.boxes)
        
    return frame

if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            frame_result = detect_object(frame)

            cv2.putText(frame_result, text, (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.imshow("Video", frame_result)
            cv2.waitKey(5)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
