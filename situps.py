


import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================
# Function to calculate angle
# ==========================
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# ==========================
# Mediapipe setup
# ==========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#  Change to your video path or 0 for webcam
video_path = r"C:\Users\woebe\situp-exercise\videos\Untitled video - Made with Clipchamp (2).mp4"
cap = cv2.VideoCapture(video_path)

counter = 0
stage = None
last_count_time = 0
angles_recorded = []  # store angles for analysis

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video processing finished or failed to load.")
            break

        # Convert frame to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

                # Calculate angle
                angle = calculate_angle(shoulder, hip, knee)
                angles_recorded.append(angle)

                # Sit-up logic with thresholds + cooldown
                if angle > 120:
                    stage = "down"
                if angle < 45 and stage == 'down':
                    if time.time() - last_count_time > 1:  # 1-sec gap
                        stage = "up"
                        counter += 1
                        last_count_time = time.time()
                        print(f" Sit-up counted! Total: {counter}")

                # Display angle on video
                cv2.putText(image, f"Angle: {int(angle)}",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", e)

        # Show counter on video
        cv2.putText(image, "Counter: " + str(counter), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show video
        cv2.imshow('Sit-Up Counter', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ==========================
# Post-analysis & Scoring
# ==========================
if angles_recorded:
    avg_angle = np.mean(angles_recorded)
    min_angle = np.min(angles_recorded)
    max_angle = np.max(angles_recorded)

    print("\n SIT-UP SESSION ANALYSIS ")
    print(f"Total Sit-ups: {counter}")
    print(f"Average torso angle: {avg_angle:.2f}°")
    print(f"Max angle reached (lying): {max_angle:.2f}°")
    print(f"Min angle reached (upright): {min_angle:.2f}°")

    # Scoring system
    if counter == 0:
        score = "No sit-ups detected"
    elif counter < 5:
        score = " Needs improvement (Beginner Level)"
    elif 5 <= counter < 15:
        score = " Good effort (Intermediate Level)"
    else:
        score = " Excellent! (Advanced Level)"

    print(f"Performance Score: {score}")

    # Form feedback
    if min_angle > 50:
        print(" Tip: Try coming up higher for a full sit-up.")
    if max_angle < 110:
        print(" Tip: Extend back more when lying down.")
    if 40 < min_angle < 50 and max_angle > 120:
        print("Great form: full range of motion!")
