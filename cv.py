import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# item_tracker = Tracker()
item_tracker = DeepSort(max_age = 30)
cam_capture = cv2.VideoCapture(0)
# yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

time_commitment = input("How long do you want this recipe to take?(e.g. 15m or 10m)")

dietary_restrictions = input("Are there any dietary restrictions?")

ingredients = set()

if not cam_capture.isOpened():
  print("Error opening camera")
  exit()

while True:
  ret, frame = cam_capture.read()

  if not ret:
    print("failed to grab frame")
    break
  results = model(frame)
  detection = results.xyxy[0]
  print(len(detection), "detection:", detection)
  # detection = detection.reshape(-1,6)
  # print(detection[0].shape)
  # print("detection:", detection)

  if detection.numel() == 0:
    print("No objects detected, skipping tracking.")
    continue
  detection = detection.cpu().numpy()
  detections_list = detection.tolist()
  items = item_tracker.update_tracks(detections_list, frame=frame)

  cv2.imshow('YOLOv5 Real-Time Detection', results.render()[0])

  for item in items:
    ingredients.add(item.track_id)
  #cv2.imshow('Webcam stream', frame)
  
  # press the q key to quit the stream
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


print("ingredients:", ingredients)
cam_capture.release()
cv2.destroyAllWindows()