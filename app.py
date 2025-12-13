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

@st.cache_resource
def setup_mediapipe():
    """Setup MediaPipe with proper error handling"""
    try:
        mp_path = Path(mp.__file__).parent
        models_src = mp_path / "modules"
        
        temp_dir_path = Path(tempfile.mkdtemp())
        models_dst = temp_dir_path / "modules"
        
        if models_src.exists() and not models_dst.exists():
            shutil.copytree(models_src, models_dst, dirs_exist_ok=True)
            for file in models_dst.rglob('*'):
                if file.is_file():
                    file.chmod(0o666)
        
        os.environ['MEDIAPIPE_MODEL_PATH'] = str(models_dst)
        
    except Exception as e:
        st.warning(f"Model setup warning: {e}")
    
    return mp.solutions.pose, mp.solutions.drawing_utils

def process_frame(image, pose, mp_pose, mp_drawing, drawing_spec, drawing_spec_points, rotate):
    """Process a single frame and return annotated image"""
    
    line_color = (255, 255, 255)
    line_color_r = (255, 0, 0)
    line_color_g = (0, 255, 0)
    line_color_b = (0, 0, 255)
    text_color = (0, 0, 0)
    
    try:
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)

        # Make a copy to avoid issues
        image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        # If no pose detected, return original image with message
        if results.pose_landmarks is None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image, "No pose detected", 
                (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            return cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

        landmarks = results.pose_landmarks.landmark

        # Get landmark coordinates
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Hide face landmarks
        for landmark_name in ['LEFT_EYE', 'RIGHT_EYE', 'LEFT_EYE_INNER', 'RIGHT_EYE_INNER', 
                             'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'NOSE', 'MOUTH_LEFT', 
                             'MOUTH_RIGHT', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']:
            landmarks[mp_pose.PoseLandmark[landmark_name].value].visibility = 0

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
        lines = [
            (left_ankle, left_knee, line_color),
            (right_ankle, right_knee, line_color_r),
            (left_hip, left_knee, line_color),
            (right_hip, right_knee, line_color_r),
            (wrist, elbow, line_color),
            (wrist_r, elbow_r, line_color_b),
            (shoulder, elbow, line_color),
            (shoulder_r, elbow_r, line_color_r),
            (shoulder, left_hip, line_color),
            (shoulder_r, right_hip, line_color_r),
        ]
        
        for p1, p2, color in lines:
            cv2.line(image, 
                    (int(p1[0] * image_width), int(p1[1] * image_height)),
                    (int(p2[0] * image_width), int(p2[1] * image_height)),
                    color, 3)

        # Draw circles
        cv2.circle(image, (int(shoulder[0] * image_width), int(shoulder[1] * image_height)), 4, line_color, 3)
        cv2.circle(image, (int(shoulder_r[0] * image_width), int(shoulder_r[1] * image_height)), 4, line_color, 3)

        # Add angle text
        angle_texts = [
            (f"knee: {left_leg_angle} / {right_leg_angle}", left_knee),
            (f"hip: {left_hip_angle} / {right_hip_angle}", left_hip),
            (f"elbow: {left_arm_angle} / {right_arm_angle}", elbow_r),
            (f"shoulder: {left_shoulder_angle} / {right_shoulder_angle}", shoulder),
        ]
        
        for text, pos in angle_texts:
            cv2.putText(
                image, text,
                (int(pos[0] * image_width - 40), int(pos[1] * image_height)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
            )

        # Convert back to BGR and draw pose landmarks
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            drawing_spec_points, connection_drawing_spec=drawing_spec
        )

        # Resize for display
        return cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
        
    except Exception as e:
        st.error(f"Error in process_frame: {str(e)}")
        # Return a blank frame in case of error
        return np.zeros((480, 640, 3), dtype=np.uint8)

def run(run_streamlit, stframe, filetype, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate):
    
    mp_pose, mp_drawing = setup_mediapipe()
    
    line_color_g = (0, 255, 0)
    line_color = (255, 255, 255)
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=line_color_g)
    drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4, color=line_color)

    if filetype == "video":
        vid = cv2.VideoCapture(input_file)
        
        if not vid.isOpened():
            st.error("Could not open video file")
            return

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.info(f"Video: {width}x{height}, {fps} fps, {total_frames} frames")
        
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, codec, fps, (width, height)) if output_file else None

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with mp_pose.Pose(
            static_image_mode=False,  # Changed to False for video
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=complexity,
            smooth_landmarks=True) as pose:
            
            frame_count = 0
            
            while vid.isOpened():
                success, image = vid.read()
                
                if not success:
                    break
                
                frame_count += 1
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Process frame
                final_frame = process_frame(
                    image, pose, mp_pose, mp_drawing, 
                    drawing_spec, drawing_spec_points, rotate
                )
                
                # Write to output if needed
                if out is not None:
                    # Resize back to original size for output
                    output_frame = cv2.resize(final_frame, (width, height))
                    out.write(output_frame)
                
                # Display in Streamlit
                if run_streamlit:
                    stframe.image(final_frame, channels="BGR")
                else:
                    cv2.imshow("Pose", final_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            
            vid.release()
            if out is not None:
                out.release()
            
            if not run_streamlit:
                cv2.destroyAllWindows()
            
            progress_bar.progress(1.0)
            status_text.text(f"Complete! Processed {frame_count} frames.")
    
    elif filetype == "image":
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
            
            image = cv2.imread(input_file)
            if image is None:
                st.error("Could not read image file")
                return
            
            # Process image
            final_frame = process_frame(
                image, pose, mp_pose, mp_drawing,
                drawing_spec, drawing_spec_points, rotate
            )
            
            # Save output
            cv2.imwrite("output_file.jpg", final_frame)
            
            if run_streamlit:
                stframe.image(final_frame, channels="BGR")
            else:
                cv2.imshow("image", final_frame)
                cv2.waitKey(0)
    else:
        st.error("ERROR in filetype")

def check_streamlit():
    """Function to check whether python code is run within streamlit"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ModuleNotFoundError:
        return False

def right(s, amount):
    return s[-amount:]

def main():
    run_streamlit = check_streamlit()
    
    if run_streamlit:
        st.header("Handstandanalyzer")
        st.write("version 131225-1345")

        detection_confidence = st.sidebar.number_input("Detection confidence", 0.0, 1.0, 0.5) 
        tracking_confidence = st.sidebar.number_input("Tracking confidence", 0.0, 1.0, 0.5) 
        complexity = st.sidebar.selectbox("Complexity", [0, 1, 2], index=1)
        rotate = st.sidebar.selectbox("Rotate", [True, False], index=1)
        
        f = st.file_uploader("Upload file (mp4 or jpg)", ['mp4', "jpg"])
        
        if f is None:
            st.info("Please upload a video or image file")
            st.stop()
        
        file_name = f.name
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{right(file_name, 3)}")
        tfile.write(f.read())
        input_file = tfile.name

        stframe = st.empty()
    else:
        detection_confidence = 0.2
        tracking_confidence = 0.2
        complexity = 0
        rotate = False
        input_file = r"C:\Users\rcxsm\Documents\python_scripts\various_projects\theo.mp4"
        file_name = input_file
        stframe = None
    
    output_file = None
    
    if right(file_name, 3) == "mp4":
        filetype = "video"
    elif right(file_name, 3) == "jpg":
        filetype = "image"
    else:
        if run_streamlit:
            st.error(f"Filetype niet herkend. [{file_name}] Upload .mp4 of .jpg")
            st.stop()
        else:
            print(f"Filetype niet herkend. [{file_name}]")
            return
    
    try:
        run(run_streamlit, stframe, filetype, input_file, output_file, 
            detection_confidence, tracking_confidence, complexity, rotate)
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        st.exception(e)

if __name__ == '__main__':
    main()