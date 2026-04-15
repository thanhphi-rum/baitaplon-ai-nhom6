from keras.models import load_model
import numpy as np
from tensorflow import keras
from imutils import face_utils
import dlib
import cv2
import face_recognition
import pygame

# ĐỊNH NGHĨA CLASS VÀ FUNCTION
"""Lưu chỉ số vị trí 2 mắt và miệng trong 68 facial landmarks"""
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    mouth_start_index, mouth_end_index = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

"""Dự đoán mắt đang mở (1) hay nhắm (0) bằng MobileNet"""
def predict_eye_state(model, image):
    image = cv2.resize(image, (20, 10))
    image = image.astype(dtype=np.float32)
    image_batch = np.reshape(image, (1, 10, 20, 1))
    image_batch = keras.applications.mobilenet.preprocess_input(image_batch)
    return np.argmax(model.predict(image_batch)[0])

"""
    Tính Mouth Aspect Ratio (MAR) để phát hiện ngáp.
    Công thức:
        MAR = (A + B + C) / (2 * D)
    Trong đó:
        A, B, C = khoảng cách dọc giữa các cặp điểm môi trên/dưới
        D       = khoảng cách ngang giữa 2 khóe miệng
    MAR > YAWN_THRESHOLD  →  đang ngáp
"""
def calculate_mar(mouth_points):
    # Khoảng cách dọc (3 cặp điểm môi ngoài)
    A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # điểm 50 - 58
    B = np.linalg.norm(mouth_points[3] - mouth_points[9])   # điểm 51 - 57
    C = np.linalg.norm(mouth_points[4] - mouth_points[8])   # điểm 52 - 56
    # Khoảng cách ngang (2 khóe miệng)
    D = np.linalg.norm(mouth_points[0] - mouth_points[6])   # điểm 48 - 54
    return (A + B + C) / (2.0 * D)

# KHỞI TẠO
"""Load model dlib nhận diện facial landmarks"""
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

"""Load model phân loại mắt mở/nhắm"""
model = load_model('weights.149-0.01.hdf5', compile=False)

"""Khởi tạo webcam"""
cap = cv2.VideoCapture(0)

# Cấu hình
SCALE           = 0.5   # thu nhỏ frame để xử lý nhanh hơn
ALARM_THRESHOLD = 5     # số frame mắt nhắm liên tiếp → báo động ngủ gật
YAWN_THRESHOLD  = 0.6   # ngưỡng MAR để xác định đang ngáp
YAWN_FRAME_MIN  = 10    # số frame ngáp liên tiếp tối thiểu → tính là 1 lần ngáp

# Biến trạng thái
count_close     = 0     # đếm frame mắt nhắm liên tiếp
count_yawn      = 0     # đếm frame đang ngáp liên tiếp
total_yawns     = 0     # tổng số lần ngáp trong phiên
alarm_playing   = False

# Khởi tạo âm thanh cảnh báo
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3')

# VÒNG LẶP CHÍNH
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize ảnh xuống 50% để tăng tốc xử lý
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=SCALE, fy=SCALE)

    # Chuyển sang LAB để lấy kênh L (Lightness) — phát hiện mặt ổn hơn dưới nhiều ánh sáng
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio = original_height / resized_height
    width_ratio  = original_width  / resized_width

    # Phát hiện khuôn mặt bằng HOG
    face_locations = face_recognition.face_locations(l, model='hog')

    if face_locations:
        # Lấy tọa độ khuôn mặt đầu tiên, chuyển về kích thước gốc
        top, right, bottom, left = face_locations[0]
        x1 = int(left  * width_ratio)
        y1 = int(top   * height_ratio)
        x2 = int(right * width_ratio)
        y2 = int(bottom * height_ratio)

        # Nhận diện 68 facial landmarks từ ảnh grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        # NHẬN DIỆN MẮT
        lx, ly, lw, lh = cv2.boundingRect(
            np.array([face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                    FacialLandMarksPosition.left_eye_end_index]])
        )
        left_eye = gray[ly:ly + lh, lx:lx + lw]

        rx, ry, rw, rh = cv2.boundingRect(
            np.array([face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                    FacialLandMarksPosition.right_eye_end_index]])
        )
        right_eye = gray[ry:ry + rh, rx:rx + rw]

        left_eye_open  = bool(predict_eye_state(model=model, image=left_eye))
        right_eye_open = bool(predict_eye_state(model=model, image=right_eye))

        # Vẽ hình chữ nhật: xanh lá = mắt mở, đỏ = mắt nhắm
        if left_eye_open and right_eye_open:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count_close = 0
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            count_close += 1

        # NHẬN DIỆN NGÁP (MAR)
        mouth_points = face_landmarks[FacialLandMarksPosition.mouth_start_index:
                                      FacialLandMarksPosition.mouth_end_index]
        mar = calculate_mar(mouth_points)

        # Vẽ đường viền miệng
        mouth_hull = cv2.convexHull(mouth_points)
        mouth_color = (0, 255, 255) if mar > YAWN_THRESHOLD else (255, 255, 0)
        cv2.drawContours(frame, [mouth_hull], -1, mouth_color, 2)

        if mar > YAWN_THRESHOLD:
            count_yawn += 1
        else:
            # Kết thúc một lần ngáp nếu đủ số frame
            if count_yawn >= YAWN_FRAME_MIN:
                total_yawns += 1
                print(f"Yawn detected! Total yawns: {total_yawns}")
            count_yawn = 0

        print(f"Left eye: {'open' if left_eye_open else 'closed'} | "
              f"Right eye: {'open' if right_eye_open else 'closed'} | "
              f"MAR: {mar:.2f} | Yawns: {total_yawns}")

    # Lật khung hình (gương)
    frame = cv2.flip(frame, 1)

    # HIỂN THỊ CẢNH BÁO
    is_sleeping = count_close > ALARM_THRESHOLD
    is_yawning  = count_yawn  >= YAWN_FRAME_MIN

    if is_sleeping:
        cv2.putText(frame, "CANH BAO: NGU GAT!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    lineType=cv2.LINE_AA)
    if is_yawning:
        cv2.putText(frame, "DANG NGAP!", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2,
                    lineType=cv2.LINE_AA)

    # Phát / tắt alarm khi phát hiện ngủ gật hoặc ngáp nhiều
    should_alarm = is_sleeping or is_yawning
    if should_alarm and not alarm_playing:
        alarm_sound.play(-1)
        alarm_playing = True
    elif not should_alarm and alarm_playing:
        alarm_sound.stop()
        alarm_playing = False

    # Hiển thị thông tin trạng thái góc trái trên
    cv2.putText(frame, f"Eyes closed: {count_close}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Total yawns: {total_yawns}", (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Sleep Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()