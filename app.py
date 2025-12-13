# Optimized for Google Cloud Run
# Lower memory usage, better temp file handling

import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
from datetime import datetime
import tempfile
import os
import shutil
from pathlib import Path
import gc

# Force read-only model loading
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "2"

# CRITICAL: Use /tmp for Cloud Run
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'

# Memory management settings
MAX_FRAMES = 300  # Limit video processing to ~10 seconds at 30fps
SKIP_FRAMES = 2   # Process every Nth frame to reduce load
RESIZE_FACTOR = 0.5  # Resize input frames to reduce memory

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

@st.cache_resource
def setup_mediapipe():
    """Setup MediaPipe for Cloud Run environment"""
    try:
        # Ensure /tmp exists and is writable
        os.makedirs('/tmp', exist_ok=True)
        
        mp_path = Path(mp.__file__).parent
        models_src = mp_path / "modules"
        
        # Use /tmp for Cloud Run
        models_dst = Path('/tmp') / 'mediapipe_modules'
        
        if models_src.exists() and not models_dst.exists():
            shutil.copytree(models_src, models_dst, dirs_exist_ok=True)
            for file in models_dst.rglob('*'):
                if file.is_file():
                    os.chmod(str(file), 0o666)
        
        os.environ['MEDIAPIPE_MODEL_PATH'] = str(models_dst)
        
    except Exception as e:
        st.warning(f"Model setup: {e}")
    
    return mp.solutions.pose, mp.solutions.drawing_utils

def process_frame(image, pose, mp_pose, mp_drawing, drawing_spec, drawing_spec_points, rotate):
    """Process a single frame with memory optimization"""
    
    line_color = (255, 255, 255)
    line_color_r = (255, 0, 0)
    line_color_g = (0, 255, 0)
    line_color_b = (0, 0, 255)
    text_color = (0, 0, 0)
    
    try:
        # Resize to reduce memory usage
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)))
        
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_180)

        # Convert color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        # Process with MediaPipe
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        # If no pose detected
        if results.pose_landmarks is None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                image, "No pose detected", 
                (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )
            # Further resize for display
            return cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

        landmarks = results.pose_landmarks.landmark

        # Get landmark coordinates (vectorized)
        def get_landmark(name):
            lm = landmarks[mp_pose.PoseLandmark[name].value]
            return [lm.x, lm.y]

        shoulder = get_landmark('LEFT_SHOULDER')
        shoulder_r = get_landmark('RIGHT_SHOULDER')
        elbow = get_landmark('LEFT_ELBOW')
        elbow_r = get_landmark('RIGHT_ELBOW')
        wrist = get_landmark('LEFT_WRIST')
        wrist_r = get_landmark('RIGHT_WRIST')
        left_hip = get_landmark('LEFT_HIP')
        right_hip = get_landmark('RIGHT_HIP')
        left_knee = get_landmark('LEFT_KNEE')
        right_knee = get_landmark('RIGHT_KNEE')
        left_ankle = get_landmark('LEFT_ANKLE')
        right_ankle = get_landmark('RIGHT_ANKLE')

        # Hide face landmarks
        face_landmarks = ['LEFT_EYE', 'RIGHT_EYE', 'LEFT_EYE_INNER', 'RIGHT_EYE_INNER', 
                         'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'NOSE', 'MOUTH_LEFT', 
                         'MOUTH_RIGHT', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']
        for lm_name in face_landmarks:
            landmarks[mp_pose.PoseLandmark[lm_name].value].visibility = 0

        # Calculate angles
        angles = {
            'left_arm': int(calculate_angle(shoulder, elbow, wrist)),
            'right_arm': int(calculate_angle(shoulder_r, elbow_r, wrist_r)),
            'left_leg': int(calculate_angle(left_hip, left_knee, left_ankle)),
            'right_leg': int(calculate_angle(right_hip, right_knee, right_ankle)),
            'left_shoulder': int(calculate_angle(left_hip, shoulder, elbow)),
            'right_shoulder': int(calculate_angle(right_hip, shoulder_r, elbow_r)),
            'left_hip': int(calculate_angle(shoulder, left_hip, left_knee)),
            'right_hip': int(calculate_angle(shoulder_r, right_hip, right_knee))
        }

        # Draw lines (vectorized)
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
                    color, 2)

        # Draw circles
        for pt in [shoulder, shoulder_r]:
            cv2.circle(image, (int(pt[0] * image_width), int(pt[1] * image_height)), 
                      3, line_color, 2)

        # Add angle text (smaller font for Cloud Run)
        font_scale = 0.5
        angle_texts = [
            (f"knee: {angles['left_leg']}/{angles['right_leg']}", left_knee),
            (f"hip: {angles['left_hip']}/{angles['right_hip']}", left_hip),
            (f"elbow: {angles['left_arm']}/{angles['right_arm']}", elbow_r),
            (f"shoulder: {angles['left_shoulder']}/{angles['right_shoulder']}", shoulder),
        ]
        
        for text, pos in angle_texts:
            cv2.putText(
                image, text,
                (int(pos[0] * image_width - 30), int(pos[1] * image_height)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA
            )

        # Convert back and draw landmarks
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            drawing_spec_points, connection_drawing_spec=drawing_spec
        )

        # Resize for display
        return cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        
    except Exception as e:
        st.error(f"Frame error: {str(e)}")
        return np.zeros((240, 320, 3), dtype=np.uint8)
    finally:
        # Force garbage collection
        gc.collect()

def run(run_streamlit, stframe, filetype, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate):
    
    mp_pose, mp_drawing = setup_mediapipe()
    
    line_color_g = (0, 255, 0)
    line_color = (255, 255, 255)
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=2, color=line_color_g)
    drawing_spec_points = mp_drawing.DrawingSpec(thickness=3, circle_radius=2, color=line_color)

    if filetype == "video":
        vid = cv2.VideoCapture(input_file)
        
        if not vid.isOpened():
            st.error("Could not open video file")
            return

        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Limit frames for Cloud Run
        max_frames = min(total_frames, MAX_FRAMES)
        
        st.warning(f"⚠️ Cloud Run mode: Processing max {max_frames} frames (every {SKIP_FRAMES}th frame)")
        st.info(f"Video: {width}x{height}, {fps} fps, {total_frames} total frames")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Use lighter model for video on Cloud Run
        with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            model_complexity=0,  # Use lightest model (0) for Cloud Run
            smooth_landmarks=True) as pose:
            
            frame_count = 0
            processed_count = 0
            
            while vid.isOpened() and processed_count < max_frames:
                success, image = vid.read()
                
                if not success:
                    break
                
                frame_count += 1
                
                # Skip frames to reduce processing
                if frame_count % SKIP_FRAMES != 0:
                    continue
                
                processed_count += 1
                
                # Update progress
                progress = processed_count / max_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {processed_count}/{max_frames}")
                
                # Process frame
                final_frame = process_frame(
                    image, pose, mp_pose, mp_drawing, 
                    drawing_spec, drawing_spec_points, rotate
                )
                
                # Display in Streamlit
                if run_streamlit:
                    stframe.image(final_frame, channels="BGR")
                
                # Free memory every 10 frames
                if processed_count % 10 == 0:
                    gc.collect()
            
            vid.release()
            progress_bar.progress(1.0)
            status_text.text(f"✅ Complete! Processed {processed_count} frames.")
            
            # Final cleanup
            gc.collect()
    
    elif filetype == "image":
        st.info("Processing image...")
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # Medium model for images
            enable_segmentation=False,  # Disable to save memory
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
            
            if run_streamlit:
                stframe.image(final_frame, channels="BGR")
                
                # Offer download
                output_path = '/tmp/output.jpg'
                cv2.imwrite(output_path, final_frame)
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Result",
                        data=f,
                        file_name="handstand_analysis.jpg",
                        mime="image/jpeg"
                    )
            
            gc.collect()
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
        st.set_page_config(page_title="Handstand Analyzer", page_icon="🤸")
        
        st.header("🤸 Handstand Analyzer")
        st.write("**Cloud Run Edition** - version 131225-1400")
        
        # Show Cloud Run tips
        with st.expander("ℹ️ Cloud Run Optimizations"):
            st.markdown("""
            - ✅ Videos limited to 300 frames (~10 sec)
            - ✅ Processing every 2nd frame
            - ✅ Images resized to 50% for memory efficiency
            - ✅ Using lightest MediaPipe model
            - ℹ️ For longer videos, consider running locally or using Cloud Storage
            """)

        detection_confidence = st.sidebar.number_input("Detection confidence", 0.0, 1.0, 0.5) 
        tracking_confidence = st.sidebar.number_input("Tracking confidence", 0.0, 1.0, 0.5) 
        rotate = st.sidebar.checkbox("Rotate 180°", False)
        
        f = st.file_uploader("Upload file (mp4 or jpg)", ['mp4', "jpg", "jpeg"])
        
        if f is None:
            st.info("👆 Please upload a video or image file")
            st.stop()
        
        file_name = f.name
        suffix = right(file_name, 3)
        
        # Save to /tmp for Cloud Run
        temp_path = f'/tmp/upload_{int(time.time())}.{suffix}'
        with open(temp_path, 'wb') as tf:
            tf.write(f.read())
        
        input_file = temp_path
        stframe = st.empty()
    else:
        detection_confidence = 0.5
        tracking_confidence = 0.5
        rotate = False
        input_file = "test.mp4"
        file_name = input_file
        stframe = None
    
    output_file = None
    
    if right(file_name, 3) in ["mp4", "MP4"]:
        filetype = "video"
    elif right(file_name, 3) in ["jpg", "peg"]:  # jpg or jpeg
        filetype = "image"
    else:
        if run_streamlit:
            st.error(f"❌ Filetype not recognized: {file_name}")
            st.stop()
        else:
            print(f"Filetype not recognized: {file_name}")
            return
    
    try:
        run(run_streamlit, stframe, filetype, input_file, output_file, 
            detection_confidence, tracking_confidence, 0, rotate)
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.exception(e)
    finally:
        # Cleanup temp files
        if run_streamlit and os.path.exists(input_file):
            try:
                os.remove(input_file)
            except:
                pass

if __name__ == '__main__':
    main()