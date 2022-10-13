from flask import Flask, render_template, Response
import cv2
import time
import pickle
import numpy as np
import mediapipe as mp
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    #function about gen_frame

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Define global variable for hand keypoints data
    hand_keypoint_data = np.array([])

    # used to record the time when we processed last frame 
    prev_frame_time = 0
    
    # used to record the time at which we processed current frame 
    new_frame_time = 0

    # Load Model
    with open("Model/svm_model_v3.sav", 'rb') as file:
        action_model = pickle.load(file)
    
    SIBI_Lang = pd.read_csv('SIBI_Lang_Spatio.csv')
    labels = SIBI_Lang['classes'].unique()


    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)

                # Draw the hand annotations on the frame.
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                

                try:
                    # Checking keypoints if complete will do this block of code
                    if len(results.multi_hand_landmarks[0].landmark) >= 21:
                        # define variable for centering and scaling process
                        centering = np.array([])
                        scaling = np.array([])

                        # Centering X coordinate Process
                        for indexPoint in range(21):
                            centering = np.append(centering, (
                                results.multi_hand_landmarks[0].landmark[indexPoint].x - results.multi_hand_landmarks[0].landmark[0].x))

                        # Centering Y coordinate Process
                        for indexPoint in range(21):
                            centering = np.append(centering, (
                                results.multi_hand_landmarks[0].landmark[indexPoint].y - results.multi_hand_landmarks[0].landmark[0].y))

                        centering = centering.reshape(2, 21)
                        
                        # Scaling Process
                        for indexIter in range(2):
                            for jointIter in range(21):
                                scaling = np.append(scaling, centering[indexIter][jointIter] / np.max(
                                np.absolute(centering[indexIter])) * 320)
                        
                        # Normalization Process
                        for jointIter in range(42):
                            hand_keypoint_data = np.append(hand_keypoint_data, (scaling[jointIter] + 320))

                        # Write spatiodata from hand keypoints coordinate
                        if len(hand_keypoint_data) >= 210:
                            # Write spatiodata to csv
                            # Uncomment 3 lines below to write hand keypoint
                            # with open('C:/Users/User/Documents/SIBI_Lang/C_Sibi_Lang.csv', 'a', newline='') as f:
                            #   writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            #   writer.writerow(hand_keypoint_data)

                            prediction = action_model.predict([hand_keypoint_data])



                            cv2.putText(frame,f'{labels[prediction[0]]}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

                            # deleted 42 old data 
                            deletedIndex = np.arange(42)
                            hand_keypoint_data = np.delete(hand_keypoint_data, deletedIndex)

                except Exception as e:
                    continue

                finally:
                    # font which we will be using to display FPS
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # time when we finish processing for this frame
                    new_frame_time = time.time()

                    fps = 1 / (new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time

                    # converting the fps into integer
                    fps = int(round(fps))

                    # converting the fps to string so that we can display it on frame
                    # by using putText function
                    fps = str(fps)

                    # puting the FPS count on the frame
                    cv2.putText(frame, fps, (550, 50), font, 2, (100, 255, 0), 3, cv2.LINE_AA)

                    # Show the result
                    cv2.imshow('Result', frame)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)