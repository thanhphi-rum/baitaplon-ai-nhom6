# Giải thích chi tiết đoạn code sleep_detect.py

Đoạn code này là một chương trình Python sử dụng thị giác máy tính để phát hiện tình trạng buồn ngủ của tài xế dựa trên việc phân tích khuôn mặt từ webcam. Nó sử dụng các thư viện như OpenCV (xử lý ảnh), dlib (nhận diện điểm đặc trưng khuôn mặt), face_recognition (phát hiện khuôn mặt), Keras (mô hình học máy), và Pygame (phát âm thanh cảnh báo). Dưới đây là giải thích chi tiết từng phần của code:

## 1. Import các thư viện (dòng 1-8)
- `from keras.models import load_model`: Tải mô hình học máy đã được huấn luyện.
- `import numpy as np`: Thư viện xử lý mảng số (dùng cho tính toán toán học).
- `from tensorflow import keras`: Framework học sâu để xử lý mô hình MobileNet.
- `from imutils import face_utils`: Công cụ tiện ích cho xử lý khuôn mặt.
- `import dlib`: Thư viện nhận diện điểm đặc trưng khuôn mặt (facial landmarks).
- `import cv2`: OpenCV cho xử lý ảnh và video.
- `import face_recognition`: Thư viện phát hiện khuôn mặt nhanh.
- `import pygame`: Phát âm thanh cảnh báo.

## 2. Định nghĩa Class và Functions (dòng 10-49)
- **Class `FacialLandMarksPosition`**: Lưu trữ chỉ số vị trí của mắt trái, mắt phải và miệng trong 68 điểm đặc trưng khuôn mặt từ dlib. Điều này giúp dễ dàng truy cập các điểm này sau.
- **Function `predict_eye_state(model, image)`**: 
  - Nhận vào một ảnh mắt (grayscale), resize về kích thước 20x10.
  - Chuyển đổi thành mảng numpy kiểu float32, reshape thành batch (1, 10, 20, 1) để phù hợp với mô hình MobileNet.
  - Tiền xử lý ảnh bằng `keras.applications.mobilenet.preprocess_input`.
  - Dự đoán trạng thái mắt (mở = 1, nhắm = 0) bằng mô hình và trả về kết quả (argmax của đầu ra).
- **Function `calculate_mar(mouth_points)`**: Tính Mouth Aspect Ratio (MAR) để phát hiện ngáp.
  - MAR = (A + B + C) / (2 * D), trong đó:
    - A, B, C: Khoảng cách dọc giữa các cặp điểm môi (điểm 50-58, 51-57, 52-56).
    - D: Khoảng cách ngang giữa 2 khóe miệng (điểm 48-54).
  - Nếu MAR > ngưỡng (YAWN_THRESHOLD), coi là đang ngáp.

## 3. Khởi tạo (dòng 51-75)
- **Load predictor dlib**: Tải mô hình `shape_predictor_68_face_landmarks.dat` để nhận diện 68 điểm đặc trưng khuôn mặt.
- **Load mô hình Keras**: Tải mô hình `weights.149-0.01.hdf5` (đã huấn luyện để phân loại mắt mở/nhắm), không compile để tránh lỗi.
- **Khởi tạo webcam**: Mở camera mặc định (index 0) bằng `cv2.VideoCapture(0)`.
- **Cấu hình hằng số**:
  - `SCALE = 0.5`: Thu nhỏ ảnh xuống 50% để xử lý nhanh hơn.
  - `ALARM_THRESHOLD = 5`: Nếu mắt nhắm liên tiếp 5 frame → cảnh báo ngủ gật.
  - `YAWN_THRESHOLD = 0.6`: Ngưỡng MAR để phát hiện ngáp.
  - `YAWN_FRAME_MIN = 10`: Cần ít nhất 10 frame liên tiếp ngáp để tính là 1 lần ngáp.
- **Biến trạng thái**: Đếm số frame mắt nhắm (`count_close`), ngáp (`count_yawn`), tổng ngáp (`total_yawns`), và trạng thái alarm.
- **Khởi tạo âm thanh**: Sử dụng Pygame để tải file `alarm.mp3` và phát lặp lại khi cần.

## 4. Vòng lặp chính (dòng 77-184)
- **Đọc frame từ webcam**: Lấy frame, nếu không thành công thì thoát.
- **Xử lý ảnh**:
  - Chuyển từ BGR sang RGB.
  - Resize xuống 50% để tăng tốc.
  - Chuyển sang không gian màu LAB, tách kênh L (độ sáng) để phát hiện mặt ổn định hơn dưới nhiều điều kiện ánh sáng.
  - Tính tỷ lệ để chuyển tọa độ về kích thước gốc.
- **Phát hiện khuôn mặt**: Sử dụng `face_recognition.face_locations` với mô hình HOG (nhanh, không cần GPU).
- **Nếu phát hiện khuôn mặt** (lấy khuôn mặt đầu tiên):
  - Chuyển tọa độ về kích thước gốc.
  - Nhận diện 68 điểm đặc trưng bằng dlib trên ảnh grayscale.
  - **Xử lý mắt**:
    - Cắt vùng mắt trái/phải từ landmarks.
    - Dự đoán trạng thái mắt bằng `predict_eye_state`.
    - Vẽ hình chữ nhật quanh mặt: xanh lá (mắt mở), đỏ (mắt nhắm).
    - Đếm frame mắt nhắm liên tiếp.
  - **Xử lý miệng (ngáp)**:
    - Lấy điểm miệng từ landmarks.
    - Tính MAR bằng `calculate_mar`.
    - Vẽ đường viền miệng: vàng (không ngáp), xanh dương (ngáp).
    - Đếm frame ngáp liên tiếp; nếu đủ ngưỡng, tăng tổng ngáp và reset đếm.
  - In trạng thái mắt, MAR, và tổng ngáp ra console.
- **Lật frame**: Để hiển thị như gương (flip ngang).
- **Hiển thị cảnh báo**:
  - Nếu mắt nhắm quá ngưỡng: Hiển thị "CANH BAO: NGU GAT!" màu đỏ.
  - Nếu đang ngáp: Hiển thị "DANG NGAP!" màu cam.
- **Phát/tắt alarm**: Phát âm thanh nếu phát hiện ngủ gật hoặc ngáp; dừng nếu hết.
- **Hiển thị thông tin**: Góc trái dưới: số frame mắt nhắm và tổng ngáp.
- **Hiển thị frame**: Trong cửa sổ "Sleep Detection".
- **Thoát**: Nhấn 'q' để dừng.

## 5. Dọn dẹp (dòng 186-187)
- Giải phóng camera và đóng cửa sổ OpenCV.

## Tóm tắt hoạt động:
Chương trình liên tục đọc từ webcam, phát hiện khuôn mặt, phân tích mắt (mở/nhắm) và miệng (ngáp), cảnh báo nếu phát hiện dấu hiệu buồn ngủ (mắt nhắm lâu hoặc ngáp nhiều). Nó sử dụng mô hình học máy để dự đoán mắt và thuật toán MAR cho miệng. Nếu cần chạy, đảm bảo có các file mô hình (`shape_predictor_68_face_landmarks.dat`, `weights.149-0.01.hdf5`, `alarm.mp3`) trong thư mục.