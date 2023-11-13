import streamlit as st
import time
import cv2 as cv
import joblib

from Buoc3.predict import visualize

app_mode = st.sidebar.selectbox('Chọn Trang',
                                ['Trang chủ',
                                 'predict',
                                 ])
# Chương trình chính

if app_mode == 'Trang chủ':
    def add_bg_from_url():
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background-image: url("data:image/jpeg;base64,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");
                 background-attachment: fixed;
                 background-size: cover
             }}
             </style>
             """,
            unsafe_allow_html=True
        )
    add_bg_from_url()
    st.balloons()
    time.sleep(1)
    st.title('ĐỒ ÁN CUỐI KÌ: Nhận diện khuôn mặt để điểm danh lớp học')
    st.markdown('THÀNH VIÊN:')
    st.title("21110309 Đoàn Huỳnh Trường Thịnh")
    st.title("21110839 Hoàng Công Mạnh")
elif app_mode == "predict":
    st.subheader('Nhận diện khuôn mặt')
    FRAME_WINDOW = st.image([])
    cap = cv.VideoCapture(0)

    svc = joblib.load('model/svc.pkl')
    mydict = ['An', 'Manh', 'Nghia', 'Phuc', 'Thinh']
if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        'model/face_detection_yunet_2022mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)

    recognizer = cv.FaceRecognizerSF.create(
        'model/face_recognition_sface_2021dec.onnx', "")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame)  # faces is a tuple
        tm.stop()

        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv.putText(frame, result, (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()