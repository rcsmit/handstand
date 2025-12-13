# Based on :
# https://raw.githubusercontent.com/Pavankunchala/Fitness-Tracking-App/main/gym_code.py
# https://github.com/Pavankunchala/Fitness-Tracking-App
# https://www.youtube.com/watch?v=bhoraBX2Dnk

import cv2
import mediapipe as mp
import argparse
import numpy as np
import time
import streamlit as st
from datetime import datetime
import tempfile
import os
import shutil
from pathlib import Path

# Force read-only model loading
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "2"

# Set writable temp directory for MediaPipe
temp_dir = tempfile.gettempdir()
os.environ['TMPDIR'] = temp_dir

@st.cache_resource
def load_mediapipe_pose():
    """Load MediaPipe pose model with caching"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True, 
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose, mp_pose

def calculate_angle(a, b, c):
    a = np.array(a)  # first
    b = np.array(b)  # mid
    c = np.array(c)  # end

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def run(run_streamlit, stframe, filetype, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate):
    line_color = (255, 255, 255)
    line_color_r = (255, 0, 0)  # used for right side
    line_color_g = (0, 255, 0)
    line_color_b = (0, 0, 255)
    text_color = (0,0,0)
    text_color_green =(0, 255, 255)

    # Copy MediaPipe models to writable temp directory
    @st.cache_resource
    def setup_mediapipe():
        try:
            # Get MediaPipe module path
            mp_path = Path(mp.__file__).parent
            models_src = mp_path / "modules"
            
            # Create temp directory for models
            temp_dir = Path(tempfile.mkdtemp())
            models_dst = temp_dir / "modules"
            
            # Copy models if they don't exist in temp
            if models_src.exists() and not models_dst.exists():
                shutil.copytree(models_src, models_dst, dirs_exist_ok=True)
                # Make all files writable
                for file in models_dst.rglob('*'):
                    if file.is_file():
                        file.chmod(0o666)
            
            # Point MediaPipe to temp location
            os.environ['MEDIAPIPE_MODEL_PATH'] = str(models_dst)
            
        except Exception as e:
            if run_streamlit:
                st.warning(f"Model setup warning: {e}")
        
        return mp.solutions.pose, mp.solutions.drawing_utils

    mp_pose, mp_drawing = setup_mediapipe()
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=(line_color_g))
    drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=(line_color))

    if filetype == "video":
        vid = cv2.VideoCapture(input_file)

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        if output_file is not None:
            out = cv2.VideoWriter(output_file, codec, fps, (width, height))
        else:
            out = None

        with mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=detection_confidence,         
            min_tracking_confidence=tracking_confidence,         
            model_complexity=complexity,    
            smooth_landmarks=True) as pose:
            
            frame_count = 0
            while vid.isOpened():
                start_time = time.time()
                success, image = vid.read()
                
                if not success:
                    break  # Exit loop when video ends
                
                frame_count += 1
                
                try:
                    final_frame = recognize_angles(
                        output_file, rotate, start_time, mp_drawing, mp_pose, 
                        calculate_angle, drawing_spec, drawing_spec_points, 
                        out, pose, image
                    )
                    
                    if final_frame is not None:
                        if run_streamlit:
                            stframe.image(final_frame, channels="BGR")
                        else:
                            cv2.imshow("Pose", final_frame)
                            if cv2.waitKey(1) & 0xFF == 27:
                                break
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    continue
            
            vid.release()
            if out is not None:
                out.release()
            
            if not run_streamlit:
                cv2.destroyAllWindows()
            else:
                st.info(f"Video processing complete. Processed {frame_count} frames.")
    
    elif filetype == "image":
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            
            image = cv2.imread(input_file)
            if image is None:
                if run_streamlit:
                    st.error("Could not read image file")
                return
                
            image = np.ascontiguousarray(image.copy())
            out = None
            start_time = time.time()
            
            try:
                image = recognize_angles(
                    output_file, rotate, start_time, mp_drawing, mp_pose, 
                    calculate_angle, drawing_spec, drawing_spec_points, 
                    out, pose, image
                )
                
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print("Succeeded")
                    cv2.imwrite("output_file.jpg", image)
                    
                    if run_streamlit:
                        stframe.image(image)
                    else:
                        cv2.imshow("image", image)
                        cv2.waitKey(0)
            except Exception as e:
                print(f"Error processing image: {e}")
                if run_streamlit:
                    st.error(f"Error processing image: {e}")
    else:
        if run_streamlit:
            st.error("ERROR in filetype")

def recognize_angles(output_file, rotate, start_time, mp_drawing, mp_pose, calculate_angle, 
                     drawing_spec, drawing_spec_points, out, pose, image):
    line_color = (255, 255, 255)
    line_color_r = (255, 0, 0)
    line_color_g = (0, 255, 0)
    line_color_b = (0, 0, 255)
    text_color = (0, 0, 0)
    text_color_green = (0, 255, 255)
    
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True

    # If no pose detected, return original image
    if results.pose_landmarks is None:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Add "No pose detected" text
        cv2.putText(
            image, "No pose detected", 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2, 
            cv2.LINE_AA
        )
        
        # Write to output if needed
        if out is not None:
            out.write(image)
        
        # Resize for display
        final_frame = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
        return final_frame

    landmarks = results.pose_landmarks.landmark

    # Get landmark coordinates
    left_eye = [
        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y
    ]
    right_eye = [
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y
    ]
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    ]
    shoulder_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    ]
    elbow = [
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
    ]
    elbow_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
    ]
    wrist = [
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    ]
    wrist_r = [
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    ]
    nose = [
        landmarks[mp_pose.PoseLandmark.NOSE.value].x, 
        landmarks[mp_pose.PoseLandmark.NOSE.value].y
    ]
    left_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    ]
    right_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    ]
    left_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    ]
    right_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
    ]
    left_ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    ]
    right_ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    ]

    # Hide face landmarks
    landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.NOSE.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0

    # Calculate angles
    left_arm_angle = int(calculate_angle(shoulder, elbow, wrist))
    right_arm_angle = int(calculate_angle(shoulder_r, elbow_r, wrist_r))
    left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))
    right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))
    left_shoulder_angle = int(calculate_angle(left_hip, shoulder, elbow))
    right_shoulder_angle = int(calculate_angle(right_hip, shoulder_r, elbow_r))
    left_hip_angle = int(calculate_angle(shoulder, left_hip, left_knee))
    right_hip_angle = int(calculate_angle(shoulder_r, right_hip, right_knee))

    # Draw lines
    cv2.line(image, (int(left_ankle[0] * image_width), int(left_ankle[1] * image_height)), 
             (int(left_knee[0] * image_width), int(left_knee[1] * image_height)), (line_color), 3)
    cv2.line(image, (int(right_ankle[0] * image_width), int(right_ankle[1] * image_height)), 
             (int(right_knee[0] * image_width), int(right_knee[1] * image_height)), (line_color_r), 3)
    cv2.line(image, (int(left_hip[0] * image_width), int(left_hip[1] * image_height)), 
             (int(left_knee[0] * image_width), int(left_knee[1] * image_height)), (line_color), 3)
    cv2.line(image, (int(right_hip[0] * image_width), int(right_hip[1] * image_height)), 
             (int(right_knee[0] * image_width), int(right_knee[1] * image_height)), (line_color_r), 3)
    cv2.line(image, (int(wrist[0] * image_width), int(wrist[1] * image_height)), 
             (int(elbow[0] * image_width), int(elbow[1] * image_height)), (line_color), 3)
    cv2.line(image, (int(wrist_r[0] * image_width), int(wrist_r[1] * image_height)), 
             (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)), (line_color_b), 3)
    cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 
             (int(elbow[0] * image_width), int(elbow[1] * image_height)), (line_color), 3)
    cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 
             (int(elbow_r[0] * image_width), int(elbow_r[1] * image_height)), (line_color_r), 3)
    cv2.line(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 
             (int(left_hip[0] * image_width), int(left_hip[1] * image_height)), (line_color), 3)
    cv2.line(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 
             (int(right_hip[0] * image_width), int(right_hip[1] * image_height)), (line_color_r), 3)

    # Draw circles
    cv2.circle(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 4, (line_color), 3)
    cv2.circle(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 4, (line_color), 3)

    # Add angle text
    cv2.putText(
        image, f"knee: {str(left_leg_angle)} / {str(right_leg_angle)}", 
        (int(left_knee[0] * image_width - 40), int(left_knee[1] * image_height)), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        image, f"hip: {str(left_hip_angle)} / {str(right_hip_angle)}", 
        (int(left_hip[0] * image_width - 40), int(left_hip[1] * image_height)), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        image, f"elbow: {str(left_arm_angle)} / {str(right_arm_angle)}", 
        (int(elbow_r[0] * image_width - 40), int(elbow_r[1] * image_height)), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
    )
    cv2.putText(
        image, f"shoulder: {str(left_shoulder_angle)} / {str(right_shoulder_angle)}", 
        (int(shoulder[0] * image_width - 40), int(shoulder[1] * image_height)), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
    )

    # Convert back to BGR and draw pose landmarks
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
        drawing_spec_points, connection_drawing_spec=drawing_spec
    )

    # Write to output file if specified
    if out is not None:
        out.write(image)

    # Resize for display
    final_frame = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
    return final_frame

def check_streamlit():
    """
    Function to check whether python code is run within streamlit
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if not get_script_run_ctx():
            use_streamlit = False
        else:
            use_streamlit = True
    except ModuleNotFoundError:
        use_streamlit = False
    return use_streamlit

def right(s, amount):
    return s[-amount:]

def main():
    run_streamlit = check_streamlit()
    
    if run_streamlit:
        st.header("Handstandanalyzer")
        st.write("version 131225-1230 na claude")

        detection_confidence = st.sidebar.number_input("Detection confidence", 0.0, 1.0, 0.5) 
        tracking_confidence = st.sidebar.number_input("Tracking confidence", 0.0, 1.0, 0.5) 
        complexity = st.sidebar.selectbox("Complexity", [0, 1, 2], index=1)
        rotate = st.sidebar.selectbox("Rotate", [True, False], index=1)
        input_file = None
    else:
        detection_confidence = 0.2
        tracking_confidence = 0.2
        complexity = 0
        rotate = False
    
    output_file = None
    
    if run_streamlit:
        f = st.file_uploader("Upload file (mp4 or jpg)", ['mp4', "jpg"])
        
        if f is not None:
            file_name = f.name
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            input_file = tfile.name
        else:
            st.stop()

        stframe = st.empty()
    else:
        input_file = r"C:\Users\rcxsm\Documents\python_scripts\various_projects\theo.mp4"
        file_name = input_file
        stframe = None
    
    if right(file_name, 3) == "mp4":
        filetype = "video"
    elif right(file_name, 3) == "jpg":
        filetype = "image"
    else:
        if run_streamlit:
            st.error(f"Filetype niet herkend. [{file_name} {right(file_name, 3)}] Upload .mp4 of .jpg")
            st.stop()
        else:
            print(f"Filetype niet herkend. [{file_name} {right(file_name, 3)}] Upload .mp4 of .jpg")
            return
        
    run(run_streamlit, stframe, filetype, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate)

if __name__ == '__main__':
    main()