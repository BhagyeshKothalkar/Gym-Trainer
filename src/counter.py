import numpy as np
import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
# from dotenv import load_dotenv
import pandas as pd
# from PIL import Image
import pickle

# load_dotenv()
model_path = 'model/mediapipe/pose_landmarker_heavy.task'
MODEL_PATH_RF = 'model/random forest/deadlift_rf.pkl'
MODEL_PATH_GB = 'model/random forest/deadlift_gb.pkl'
# VIDEO_PATH = '/data/videos/deadlift/deadlift_24.mp4'
VIDEO_PATH = 'data/videos/deadlift/deadlift_24.mp4'
MAX_COMPARE_FOR_STATE = 3
angle_thresholds = {
    "neck": 10,              # head tilt / forward neck
    "left_elbow": 15,        # elbow flexion symmetry & control
    "right_elbow": 15,

    "left_shoulder": 12,     # shoulder elevation / swing
    "right_shoulder": 12,

    "left_hip": 10,          # torso lean / hip hinge
    "right_hip": 10,

    "left_knee": 8,          # knee lock or bend
    "right_knee": 8,

    "left_ankle": 8,         # foot stability
    "right_ankle": 8
}


def calculateAngle(a, b, c):
  a = np.array(a)  # 첫 번째 지점
  b = np.array(b)  # 중간 지점
  c = np.array(c)  # 끝 지점

  radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
      a[1] - b[1], a[0] - b[0]
  )
  angle = np.abs(radians * 180.0 / np.pi)

  if angle > 180.0:
      angle = 360 - angle

  return angle

def angles(pose_landmarker_result):
    nose = [
    pose_landmarker_result[0][0].x,
    pose_landmarker_result[0][0].y,
    ]   
    left_shoulder = [
        pose_landmarker_result[0][11].x,
        pose_landmarker_result[0][11].y,
    ]  # 좌측 어깨
    left_elbow = [
        pose_landmarker_result[0][13].x,
        pose_landmarker_result[0][13].y,
    ]  # 좌측 팔꿈치
    left_wrist = [
        pose_landmarker_result[0][15].x,
        pose_landmarker_result[0][15].y,
    ]  # 좌측 손목
    left_hip = [
        pose_landmarker_result[0][23].x,
        pose_landmarker_result[0][23].y,
    ]  # 좌측 힙
    left_knee = [
        pose_landmarker_result[0][25].x,
        pose_landmarker_result[0][25].y,
    ]  # 좌측 무릎
    left_ankle = [
        pose_landmarker_result[0][27].x,
        pose_landmarker_result[0][27].y,
    ]  # 좌측 발목
    left_heel = [
        pose_landmarker_result[0][29].x,
        pose_landmarker_result[0][29].y,
    ]  # 좌측 힐
    right_shoulder = [
        pose_landmarker_result[0][12].x,
        pose_landmarker_result[0][12].y,
    ]  # 우측 어깨
    right_elbow = [
        pose_landmarker_result[0][14].x,
        pose_landmarker_result[0][14].y,
    ]  # 우측 팔꿈치
    right_wrist = [
        pose_landmarker_result[0][16].x,
        pose_landmarker_result[0][16].y,
    ]  # 우측 손목
    right_hip = [
        pose_landmarker_result[0][24].x,
        pose_landmarker_result[0][24].y,
    ]  # 우측 힙
    right_knee = [
        pose_landmarker_result[0][26].x,
        pose_landmarker_result[0][26].y,
    ]  # 우측 무릎
    right_ankle = [
        pose_landmarker_result[0][28].x,
        pose_landmarker_result[0][28].y,
    ]  # 우측 발목
    right_heel = [
        pose_landmarker_result[0][30].x,
        pose_landmarker_result[0][30].y,
    ]

    neck_angle = (
        calculateAngle(left_shoulder, nose, left_hip)
        + calculateAngle(right_shoulder, nose, right_hip) / 2
    )
    left_elbow_angle = calculateAngle(
        left_shoulder, left_elbow, left_wrist
    )
    right_elbow_angle = calculateAngle(
        right_shoulder, right_elbow, right_wrist
    )
    left_shoulder_angle = calculateAngle(
        left_elbow, left_shoulder, left_hip
    )
    right_shoulder_angle = calculateAngle(
        right_elbow, right_shoulder, right_hip
    )
    left_hip_angle = calculateAngle(
        left_shoulder, left_hip, left_knee
    )
    right_hip_angle = calculateAngle(
        right_shoulder, right_hip, right_knee
    )
    left_knee_angle = calculateAngle(
        left_hip, left_knee, left_ankle
    )
    right_knee_angle = calculateAngle(
        right_hip, right_knee, right_ankle
    )
    left_ankle_angle = calculateAngle(
        left_knee, left_ankle, left_heel
    )
    right_ankle_angle = calculateAngle(
        right_knee, right_ankle, right_heel
    )
    # print("neck_angle: ", neck_angle)
    # print("left_elbow_angle: ", left_elbow_angle)
    # print("right_elbow_angle: ", right_elbow_angle)
    # print("left_shoulder_angle: ", left_shoulder_angle)
    # print("right_shoulder_angle: ", right_shoulder_angle)
    # print("left_hip_angle: ", left_hip_angle)
    # print("right_hip_angle: ", right_hip_angle)
    # print("left_knee_angle: ", left_knee_angle)
    # print("right_knee_angle: ", right_knee_angle)
    # print("left_ankle_angle: ", left_ankle_angle)
    # print("right_ankle_angle: ", right_ankle_angle)
    return {"neck":neck_angle,
            "left_elbow":left_elbow_angle,
            "right_elbow":right_elbow_angle,
            "left_shoulder":left_shoulder_angle,
            "right_shoulder":right_shoulder_angle,
            "left_hip":left_hip_angle,
            "right_hip":right_hip_angle,
            "left_knee":left_knee_angle,
            "right_knee":right_knee_angle,
            "left_ankle":left_ankle_angle,
            "right_ankle":right_ankle_angle
            }
    # angles_list = ["neck", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_hip, right_shoulder"]

def show_landmark(pose_landmarker_result, frame_rgb, window_name = 'Display'):
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarker_result[0]
    ])
    annoted = np.copy(frame_rgb)
    mp.solutions.drawing_utils.draw_landmarks(
    annoted,
    pose_landmarks_proto,
    mp.solutions.pose.POSE_CONNECTIONS,
    mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
    mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2),
    )
    cv2.imshow(window_name, annoted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

landmarker = PoseLandmarker.create_from_options(options)

with open(MODEL_PATH_RF, "rb") as f:
    model_e = pickle.load(f)


def calculate_state(state_history:list)->str | None:
    if len(state_history) >= MAX_COMPARE_FOR_STATE:
        if all(ite=='up' for ite in state_history[-MAX_COMPARE_FOR_STATE:]):
            return 'up'
        elif all(ite=='down' for ite in state_history[-MAX_COMPARE_FOR_STATE:]):
            return 'down'
    return None


video = cv2.VideoCapture(VIDEO_PATH)
counter = 0
posture_state = [None]
while True:
    ret, frame = video.read()
    if not ret:
        print('Ending')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_image)
    results_pose = landmarker.detect(mp_image)

    if len(results_pose.pose_landmarks) == 0 : continue
    row = [
    coord
    for res in results_pose.pose_landmarks[0]
    for coord in [res.x, res.y, res.z, res.visibility]
    ]
    X = pd.DataFrame([row])
    exercise_class = model_e.predict(X)[0]
    exercise_class_prob = model_e.predict_proba(X)[0]
    print(exercise_class)
    if 'up' in exercise_class:
        prev_state = calculate_state(posture_state)
        if (prev_state=='down'): counter+=1
        posture_state.append('up')
    elif 'down' in exercise_class:
        prev_state = calculate_state(posture_state)
        if (prev_state=='up'): counter+=1
        posture_state.append('down')

    show_landmark(results_pose.pose_landmarks, frame_rgb, 'display')
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
print(counter/2)
print(posture_state)




# print(dir(pose_landmarker_result))
# print(type(pose_landmarker_result.pose_landmarks[0]))
# print(dir(pose_landmarker_result.pose_landmarks))
# print(pose_landmarker_incorrect.pose_landmarks[0][0].x)

