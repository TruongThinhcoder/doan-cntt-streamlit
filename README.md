*/ Dự án Nhận Diện Khuôn Mặt /*
* Tổng Quan
Dự án này là một ứng dụng nhận diện khuôn mặt sử dụng OpenCV và Streamlit. Nó sử dụng các mô hình được huấn luyện trước cho việc nhận diện và nhận biết khuôn mặt để xác định các khuôn mặt trong ảnh hoặc qua luồng video trực tiếp từ webcam.

* Tính Năng
- Nhận Diện Khuôn Mặt: Sử dụng mô hình nhận diện khuôn mặt để xác định vị trí của khuôn mặt trong ảnh hoặc khung hình video.
- Nhận Biết Khuôn Mặt: Nhận biết khuôn mặt bằng cách so sánh chúng với cơ sở dữ liệu của những người đã biết trước.
- Xử lý Video Thời Gian Thực: Cho phép nhận diện khuôn mặt trực tiếp từ webcam.
* Cài Đặt
1. Clone Dự Án:
git clone: https://github.com/TruongThinhcoder/doan-cntt-streamlit.git
cd doan-cntt-streamlit
2. Cài Đặt Thư Viện:
pip install -r requirements.txt
3. Sử Dụng
Chạy Ứng Dụng: streamlit run FinalCNTT.py
Điều này sẽ mở ứng dụng Streamlit, nơi bạn có thể tương tác với hệ thống nhận diện khuôn mặt.

* Hướng Dẫn:
Chọn tùy chọn phù hợp cho ảnh hoặc video đầu vào, hoặc chọn webcam để nhận diện khuôn mặt thời gian thực.
Tuân theo các hướng dẫn trên màn hình để ứng dụng nhận diện và nhận biết khuôn mặt.
* Mô Hình và Tài Nguyên
Mô Hình Nhận Diện Khuôn Mặt: Tải từ OpenCV Model Zoo
Mô Hình Nhận Biết Khuôn Mặt: Tải từ OpenCV Model Zoo
