import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from line import Line
from point import Point

#for getting landmarks
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

#for getting ms
start = time.perf_counter()

#for drawing landmarks
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#result from landmarks
latest_result = None
latest_frame = None
main_line: Line

def main():
    print("hello")

def get_ms():
    return int((time.perf_counter() - start) * 1000)

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # to print results 
    # print('hand landmarker result: {}'.format(result))
    global latest_result, main_line
    latest_result = result

    process_line(result)
    
def process_line(result: HandLandmarkerResult):
    global main_line
    lm = result.hand_landmarks[0]
    thumb_point = Point(lm[3].x, lm[3].y, lm[3].z)
    pointer_point = Point(lm[7].x, lm[7].y, lm[7].z)
    if main_line is None:
        handedness = result.handedness[0].category_name
        h = False
        if handedness == "right hand":
            h = True
        main_line = Line(thumb_point, pointer_point, h)
    else:
        main_line.change_line(thumb_point, pointer_point)

def camera():
    global latest_result
    cap = cv2.VideoCapture(1)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        #num_hands=2,
        result_callback=save_result)
    
    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image,1)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            ms = get_ms()
            landmarker.detect_async(image, ms)
            

            image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

            if latest_result is not None:
                if latest_result.hand_landmarks:
                    image = draw_landmarks_on_image(image, latest_result)
            cv2.imshow("Hand Tracking", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == "__main__":
    camera()