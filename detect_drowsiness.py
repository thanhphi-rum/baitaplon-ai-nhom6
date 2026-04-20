# ===== IMPORT CÁC THƯ VIỆN CẦN THIẾT =====
from keras.models import load_model          # Load mô hình đã train
import numpy as np                           # Xử lý mảng và toán học
from tensorflow import keras                 # TensorFlow/Keras framework
from imutils import face_utils               # Hỗ trợ xử lý facial landmarks
import dlib                                  # Nhận dạng khuôn mặt và landmarks
import cv2                                   # OpenCV - xử lý hình ảnh video
import face_recognition                      # Phát hiện khuôn mặt
import pygame                                # Phát âm thanh báo động


# ===== CẤU HÌNH HỆ THỐNG =====
SCALE = 0.5             # Tỷ lệ thu nhỏ frame (giúp xử lý nhanh hơn)
ALARM_THRESHOLD = 5     # Số frame mắt nhắm liên tiếp để kích hoạt cảnh báo
YAWN_THRESHOLD = 0.6    # Ngưỡng MAR (Mouth Aspect Ratio) để xác định ngáp
YAWN_FRAME_MIN = 10     # Số frame ngáp liên tiếp tối thiểu để tính là 1 lần ngáp


# ===== CLASS LƯU TRỮ VỊ TRÍ CÁC ĐIỂM TRÊN KHUÔN MẶT =====
# Mô hình dlib sử dụng 68 điểm để xác định các vùng trên khuôn mặt
class FacialLandMarksPosition:
    # Chỉ số các điểm đặc trưng trong 68 điểm của mắt trái và phải
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # Chỉ số các điểm đặc trưng của miệng
    mouth_start_index, mouth_end_index = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def predict_eye_state(model, image):
    """
    Dự đoán trạng thái mắt từ hình ảnh vùng mắt.
    Trả về: 1 (mắt mở) hoặc 0 (mắt nhắm)
    """
    # Resize ảnh mắt về kích thước chuẩn 20x10 pixels
    image = cv2.resize(image, (20, 10)).astype(np.float32)
    
    # Chuẩn bị ảnh cho mô hình MobileNet (normalize giá trị pixel)
    image_batch = keras.applications.mobilenet.preprocess_input(
        np.reshape(image, (1, 10, 20, 1))  # Shape: (batch, height, width, channels)
    )
    
    # Dự đoán và lấy lớp có xác suất cao nhất
    return np.argmax(model.predict(image_batch)[0])


def calculate_mar(mouth_points):
    """
    Tính Mouth Aspect Ratio (MAR) - chỉ số đo độ mở miệng.
    Công thức: MAR = (A + B + C) / (2 * D)
    - A, B, C: Khoảng cách dọc (chiều cao miệng) tại 3 vị trí
    - D: Khoảng cách ngang (chiều rộng miệng)
    - MAR cao → miệng mở rộng (đang ngáp)
    """
    # Tính 3 khoảng cách dọc từ các điểm landmarks của miệng
    A = np.linalg.norm(mouth_points[2] - mouth_points[10])
    B = np.linalg.norm(mouth_points[3] - mouth_points[9])
    C = np.linalg.norm(mouth_points[4] - mouth_points[8])
    
    # Tính khoảng cách ngang
    D = np.linalg.norm(mouth_points[0] - mouth_points[6])
    
    # Trả về chỉ số MAR
    return (A + B + C) / (2.0 * D)


def get_eye_region(gray, landmarks, start, end):
    """
    Cắt vùng mắt từ ảnh xám (grayscale) dựa trên facial landmarks.
    Tham số:
    - gray: Ảnh xám
    - landmarks: Danh sách 68 điểm đặc trưng
    - start, end: Chỉ số bắt đầu và kết thúc trong danh sách landmarks
    Trả về: Ảnh vùng mắt được cắt
    """
    # Lấy hình chữ nhật bounding box của vùng mắt
    x, y, w, h = cv2.boundingRect(np.array([landmarks[start:end]]))
    # Cắt vùng mắt từ ảnh
    return gray[y:y + h, x:x + w]


# ===== KHỞI TẠO MÔ HÌNH VÀ CÁC THÀNH PHẦN =====
# Load mô hình dlib để nhận dạng 68 facial landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load mô hình đã train để phân loại trạng thái mắt (mở/nhắm)
model = load_model('weights.149-0.01.hdf5', compile=False)

# Khởi tạo webcam (camera mặc định)
cap = cv2.VideoCapture(0)

# Khởi tạo hệ thống âm thanh và tải âm thanh báo động
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3')

# ===== BIẾN TRẠNG THÁI =====
count_close = 0     # Đếm số frame mắt nhắm liên tiếp
count_yawn = 0      # Đếm số frame đang ngáp liên tiếp
total_yawns = 0     # Tổng số lần ngáp trong toàn bộ phiên
alarm_playing = False  # Cờ theo dõi xem âm thanh có đang phát không

# ===== VÒNG LẶP CHÍNH - XỬ LÝ TỪNG FRAME VIDEO =====
while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]  # Lưu kích thước gốc
    
    # Thu nhỏ ảnh để tăng tốc độ xử lý
    resized = cv2.resize(image, (0, 0), fx=SCALE, fy=SCALE)

    # Chuyển ảnh sang không gian màu LAB, lấy channel L (độ sáng)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)  # Tách channel L để dùng cho nhận dạng khuôn mặt
    res_h, res_w = l.shape[:2]
    
    # Tính tỷ lệ kích thước để chuyển đổi tọa độ từ ảnh thu nhỏ về gốc
    h_ratio = orig_h / res_h
    w_ratio = orig_w / res_w

    # Phát hiện khuôn mặt trong ảnh thu nhỏ bằng phương pháp HOG (Histogram of Oriented Gradients)
    face_locations = face_recognition.face_locations(l, model='hog')

    if face_locations:  # Nếu phát hiện được khuôn mặt
        # Chuyển đổi tọa độ từ ảnh thu nhỏ sang ảnh gốc
        top, right, bottom, left = face_locations[0]
        x1, y1 = int(left * w_ratio), int(top * h_ratio)
        x2, y2 = int(right * w_ratio), int(bottom * h_ratio)

        # Chuyển frame sang ảnh xám để xử lý landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Nhận dạng 68 facial landmarks trong vùng khuôn mặt
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        # Chuyển đổi dữ liệu landmarks sang numpy array
        landmarks = face_utils.shape_to_np(shape)

        # Trích xuất các vùng mắt từ ảnh xám
        p = FacialLandMarksPosition
        left_eye = get_eye_region(gray, landmarks, p.left_eye_start_index, p.left_eye_end_index)
        right_eye = get_eye_region(gray, landmarks, p.right_eye_start_index, p.right_eye_end_index)

        # Dự đoán trạng thái của mỗi mắt (1=mở, 0=nhắm)
        left_open = bool(predict_eye_state(model, left_eye))
        right_open = bool(predict_eye_state(model, right_eye))

        # Kiểm tra trạng thái mắt
        if left_open and right_open:
            # Cả 2 mắt mở → vẽ khung xanh, đặt lại bộ đếm
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            count_close = 0
        else:
            # Ít nhất 1 mắt nhắm → vẽ khung đỏ, tăng bộ đếm
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            count_close += 1

        # ===== XỬ LÝ PHÁT HIỆN NGÁP =====
        # Lấy các điểm landmarks của miệng
        mouth_pts = landmarks[p.mouth_start_index:p.mouth_end_index]
        
        # Tính chỉ số MAR (độ mở miệng)
        mar = calculate_mar(mouth_pts)

        # Vẽ đường viền miệng trên frame
        mouth_hull = cv2.convexHull(mouth_pts)
        # Chọn màu: vàng nhạt (đang ngáp) hoặc vàng đậm (không ngáp)
        mouth_color = (0, 255, 255) if mar > YAWN_THRESHOLD else (255, 255, 0)
        cv2.drawContours(frame, [mouth_hull], -1, mouth_color, 2)

        # Cập nhật bộ đếm ngáp
        if mar > YAWN_THRESHOLD:
            # Đang ngáp (miệng mở rộng)
            count_yawn += 1
        else:
            # Miệng vừa đóng lại
            # Nếu trước đó ngáp đủ lâu (>= YAWN_FRAME_MIN), tính là 1 lần ngáp
            if count_yawn >= YAWN_FRAME_MIN:
                total_yawns += 1
                print(f"Yawn detected! Total yawns: {total_yawns}")
            count_yawn = 0

        # In thông tin debug
        print(
            f"Left: {'open' if left_open else 'closed'} | "
            f"Right: {'open' if right_open else 'closed'} | "
            f"MAR: {mar:.2f} | Yawns: {total_yawns}"
        )

    # Lật frame theo chiều dọc (mirror effect)
    frame = cv2.flip(frame, 1)

    # ===== KIỂM TRA ĐIỀU KIỆN BÁO ĐỘNG =====
    # Xác định xem tài xế có đang ngủ gật hoặc ngáp không
    is_sleeping = count_close > ALARM_THRESHOLD
    is_yawning = count_yawn >= YAWN_FRAME_MIN

    # Hiển thị cảnh báo ngủ gật
    if is_sleeping:
        cv2.putText(frame, "CANH BAO: NGU GAT!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    # Hiển thị cảnh báo ngáp
    if is_yawning:
        cv2.putText(frame, "DANG NGAP!", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)

    # ===== KIỂM SOÁT ÂM THANH BÁO ĐỘNG =====
    # Nên phát âm thanh nếu phát hiện ngủ gật hoặc ngáp
    should_alarm = is_sleeping or is_yawning
    
    if should_alarm and not alarm_playing:
        # Bắt đầu phát âm thanh báo động (loop vô hạn)
        alarm_sound.play(-1)
        alarm_playing = True
    elif not should_alarm and alarm_playing:
        # Dừng phát âm thanh
        alarm_sound.stop()
        alarm_playing = False

    # ===== HIỂN THỊ THÔNG TIN TRÊN MÀNG HÌNH =====
    # Hiển thị số frame mắt nhắm
    cv2.putText(frame, f"Eyes closed: {count_close}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    # Hiển thị tổng số lần ngáp
    cv2.putText(frame, f"Total yawns: {total_yawns}", (10, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Hiển thị frame xử lý
    cv2.imshow('Sleep Detection', frame)
    
    # Kiểm tra phím nhấn, thoát nếu nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== DỌN DẸP VÀ ĐÓNG CHƯƠNG TRÌNH =====
cap.release()  # Giải phóng camera
cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV