# pip install mediapipe opencv-python
import cv2
import math
import mediapipe as mp

def angle_between_vectors(v1: tuple[float, float],
                          v2: tuple[float, float]) -> float:
    """
    Compute signed angle (in degrees) between 2D vectors v1 → v2.
    Positive = counter‐clockwise rotation, negative = clockwise.
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag == 0:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot/mag))
    ang = math.degrees(math.acos(cos_val))
    cross_z = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross_z >= 0 else -ang

def landmark_to_pixel(landmark, fw: int, fh: int) -> tuple[int, int]:
    """Normalized landmark → pixel coords."""
    return int(landmark.x*fw), int(landmark.y*fh)

def RULA_calculator(shoulder_angle: float, elbow_angle: float) -> tuple[int,int]:
    """
    Map angles (0° = arm down) into RULA scores:
      Upper arm: ≤20°→1, ≤45°→2, ≤90°→3, >90°→4
      Lower arm: 60–100°→1, >100°→3, otherwise→2
    """
    sa = abs(shoulder_angle)
    if   sa <= 20:  up = 1
    elif sa <= 45:  up = 2
    elif sa <= 90:  up = 3
    else:           up = 4

    if   60 <= elbow_angle <= 100: low = 1
    elif elbow_angle > 100:        low = 3
    else:                           low = 2

    return up, low

def main():
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = pose.process(rgb)
        rgb.flags.writeable = True
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = res.pose_landmarks.landmark
            down = (0.0, 1.0)  # reference vector pointing straight down

            # RIGHT ARM
            sh_r = landmark_to_pixel(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], w, h)
            el_r = landmark_to_pixel(lm[mp_pose.PoseLandmark.RIGHT_ELBOW],    w, h)
            wr_r = landmark_to_pixel(lm[mp_pose.PoseLandmark.RIGHT_WRIST],    w, h)
            v_sh_el = (el_r[0] - sh_r[0], el_r[1] - sh_r[1])
            v_el_wr = (wr_r[0] - el_r[0], wr_r[1] - el_r[1])
            shoulder_r = angle_between_vectors(v_sh_el, down)
            shoulder_r = -shoulder_r
            elbow_r    = abs(angle_between_vectors(v_el_wr, down))

            # LEFT ARM
            sh_l = landmark_to_pixel(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], w, h)
            el_l = landmark_to_pixel(lm[mp_pose.PoseLandmark.LEFT_ELBOW],    w, h)
            wr_l = landmark_to_pixel(lm[mp_pose.PoseLandmark.LEFT_WRIST],    w, h)
            v_sh_el_l = (el_l[0] - sh_l[0], el_l[1] - sh_l[1])
            v_el_wr_l = (wr_l[0] - el_l[0], wr_l[1] - el_l[1])
            shoulder_l = angle_between_vectors(v_sh_el_l, down)  # flip sign
            elbow_l    = abs(angle_between_vectors(v_el_wr_l, down))

            # Compute RULA scores
            up_r, low_r = RULA_calculator(shoulder_r, elbow_r)
            up_l, low_l = RULA_calculator(shoulder_l, elbow_l)

            # Overlay raw angles
            cv2.putText(out, f"ShR: {shoulder_r:.1f}°", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(out, f"ElR: {elbow_r:.1f}°",    (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(out, f"ShL: {shoulder_l:.1f}°", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.putText(out, f"ElL: {elbow_l:.1f}°",    (10,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Overlay RULA scores with new labels
            cv2.putText(out, f"Shoulder RULA L: {up_l}", (250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(out, f"Shoulder RULA R: {up_r}", (250, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(out, f"Elbow RULA L:    {low_l}", (250, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(out, f"Elbow RULA R:    {low_r}", (250,120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow('RULA Pose', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
