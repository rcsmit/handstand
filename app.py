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
from collections import defaultdict

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

# Ideal handstand angles (in degrees)
IDEAL_ANGLES = {
    'shoulder': 180,
    'elbow': 180,
    'hip': 180,
    'knee': 180
}

# Weights for scoring (must sum to 1.0)
WEIGHTS = {
    'shoulder': 0.30,  # Most important for form
    'elbow': 0.25,     # Safety critical
    'hip': 0.30,       # Body line
    'knee': 0.15       # Less critical
}

COMBINED_FACTOR = 0.85 # combined = (total_score * COMBINED_FACTOR + symmetry_score * (1-COMBINED_FACTOR))

def calculate_joint_score(actual_angle, ideal_angle):
    """Calculate score for a single joint (0-100)"""
    deviation = abs(actual_angle - ideal_angle)
    # Linear decay: perfect at 0° deviation, 0 at 50° deviation
    score = max(0, 100 - (deviation * 2))
    return score

def calculate_handstand_score(angles):
    """
    Calculate weighted handstand score
    angles = {
        'left_shoulder': 175, 'right_shoulder': 178,
        'left_elbow': 180, 'right_elbow': 179,
        'left_hip': 165, 'right_hip': 168,
        'left_knee': 180, 'right_knee': 180
    }
    """
    scores = {}
    
    # Calculate average for each joint (left + right)
    for joint in ['shoulder', 'elbow', 'hip', 'knee']:
        left_angle = angles.get(f'left_{joint}', 0)
        right_angle = angles.get(f'right_{joint}', 0)
        avg_angle = (left_angle + right_angle) / 2
        
        scores[joint] = calculate_joint_score(avg_angle, IDEAL_ANGLES[joint])
    
    # Calculate weighted total score
    total_score = sum(scores[joint] * WEIGHTS[joint] for joint in scores)
    
    return total_score, scores

def calculate_symmetry_score(angles):
    """Calculate left/right symmetry score"""
    symmetry_scores = {}
    
    for joint in ['shoulder', 'elbow', 'hip', 'knee']:
        left = angles.get(f'left_{joint}', 0)
        right = angles.get(f'right_{joint}', 0)
        diff = abs(left - right)
        
        # Perfect symmetry = 100, decreases by 3 points per degree difference
        symmetry_scores[joint] = max(0, 100 - (diff * 3))
    
    avg_symmetry = sum(symmetry_scores.values()) / len(symmetry_scores)
    return avg_symmetry, symmetry_scores

def generate_feedback(angles, form_scores, symmetry_scores):
    """Generate actionable feedback based on angles"""
    feedback = []
    
    # Check shoulders
    avg_shoulder = (angles['left_shoulder'] + angles['right_shoulder']) / 2
    if avg_shoulder < 170:
        feedback.append("💡 Open your shoulders more - push through your hands and reach tall")
    elif avg_shoulder < 175:
        feedback.append("👍 Good shoulder angle - try to open them a bit more")
    else:
        feedback.append("✨ Excellent shoulder position!")
    
    # Check elbows
    avg_elbow = (angles['left_elbow'] + angles['right_elbow']) / 2
    if avg_elbow < 170:
        feedback.append("⚠️ Lock your elbows! Bent arms are dangerous and unstable")
    elif avg_elbow < 178:
        feedback.append("💪 Almost there - lock out those elbows completely")
    else:
        feedback.append("✨ Perfect elbow lock!")
    
    # Check hips
    avg_hip = (angles['left_hip'] + angles['right_hip']) / 2
    if avg_hip < 165:
        feedback.append("💡 Squeeze your glutes and engage your core to straighten your body")
    elif avg_hip < 175:
        feedback.append("👍 Good body line - focus on staying straight")
    else:
        feedback.append("✨ Excellent straight body line!")
    
    # Check knees
    avg_knee = (angles['left_knee'] + angles['right_knee']) / 2
    if avg_knee < 170:
        feedback.append("💡 Straighten your legs - point your toes")
    elif avg_knee < 178:
        feedback.append("👍 Almost straight - lock those knees")
    else:
        feedback.append("✨ Perfect leg extension!")
    
    # Check symmetry
    for joint in ['shoulder', 'elbow', 'hip', 'knee']:
        if symmetry_scores[joint] < 85:
            left = angles[f'left_{joint}']
            right = angles[f'right_{joint}']
            side = "left" if left < right else "right"
            feedback.append(f"⚖️ {joint.capitalize()} asymmetry - your {side} side needs work")
    
    return feedback

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

def process_frame(image, pose, mp_pose, mp_drawing, drawing_spec, drawing_spec_points, rotate, return_angles=False):
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
            return (cv2.resize(image, (0, 0), fx=0.4, fy=0.4), None) if return_angles else cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

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
            'right_hip': int(calculate_angle(shoulder_r, right_hip, right_knee)),
            'left_elbow': int(calculate_angle(shoulder, elbow, wrist)),
            'right_elbow': int(calculate_angle(shoulder_r, elbow_r, wrist_r)),
            'left_knee': int(calculate_angle(left_hip, left_knee, left_ankle)),
            'right_knee': int(calculate_angle(right_hip, right_knee, right_ankle))
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
            (f"knee: {angles['left_knee']}/{angles['right_knee']}", left_knee),
            (f"hip: {angles['left_hip']}/{angles['right_hip']}", left_hip),
            (f"elbow: {angles['left_elbow']}/{angles['right_elbow']}", elbow_r),
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
        final_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        
        if return_angles:
            return final_frame, angles
        return final_frame
        
    except Exception as e:
        st.error(f"Frame error: {str(e)}")
        result = np.zeros((240, 320, 3), dtype=np.uint8)
        return (result, None) if return_angles else result
    finally:
        # Force garbage collection
        gc.collect()

def run(run_streamlit, stframe, filetype, input_file, output_file, detection_confidence, tracking_confidence, complexity, rotate, show_live=True):
    
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
        
        # Create output video file
        output_path = '/tmp/output_video.mp4'
        output_width = int(width * RESIZE_FACTOR * 0.5)  # Display size
        output_height = int(height * RESIZE_FACTOR * 0.5)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps // SKIP_FRAMES, 
                                       (output_width, output_height))
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Collect angles for scoring
        all_angles = []
        
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
                result = process_frame(
                    image, pose, mp_pose, mp_drawing, 
                    drawing_spec, drawing_spec_points, rotate, return_angles=True
                )
                final_frame, frame_angles = result
                
                # Collect angles for scoring
                if frame_angles:
                    all_angles.append(frame_angles)
                
                # Write to output video
                video_writer.write(final_frame)
                
                # Display in Streamlit (only show every 5th frame to reduce load)
                if run_streamlit and show_live and processed_count % 5 == 0:
                    stframe.image(final_frame, channels="BGR", use_container_width=True)
                    time.sleep(0.01)
                
                # Store last frame for batch mode
                last_frame = final_frame
                
                # Free memory every 10 frames
                if processed_count % 10 == 0:
                    gc.collect()
            
            vid.release()
            video_writer.release()
            progress_bar.progress(1.0)
            status_text.text(f"✅ Complete! Processed {processed_count} frames.")
            
            # Calculate scores if we have angle data
            if all_angles:
                # Average angles across all frames
                avg_angles = {}
                for key in all_angles[0].keys():
                    avg_angles[key] = sum(frame[key] for frame in all_angles) / len(all_angles)
                
                # Calculate scores
                total_score, form_scores = calculate_handstand_score(avg_angles)
                symmetry_score, symmetry_scores = calculate_symmetry_score(avg_angles)
                feedback = generate_feedback(avg_angles, form_scores, symmetry_scores)
                
                # Display scores
                st.subheader("🏆 Handstand Analysis")
                
                # Overall score with visual
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Form Score", f"{total_score:.1f}/100")
                with col2:
                    st.metric("Symmetry Score", f"{symmetry_score:.1f}/100")
                with col3:
<<<<<<< HEAD
                    # combined = (total_score * 0.7 + symmetry_score * 0.3)
                    combined = (total_score * COMBINED_FACTOR + symmetry_score * (1-COMBINED_FACTOR))
                    
=======
                    combined = (total_score * COMBINED_FACTOR + symmetry_score * (1-COMBINED_FACTOR))
>>>>>>> 3dc0149327b850c90a577767dfd2a87e36cebbf3
                    st.metric("Overall Score", f"{combined:.1f}/100")
                
                # Grade
                if combined >= 90:
                    grade = "⭐⭐⭐⭐⭐ EXCELLENT"
                    grade_color = "green"
                elif combined >= 80:
                    grade = "⭐⭐⭐⭐ GREAT"
                    grade_color = "blue"
                elif combined >= 70:
                    grade = "⭐⭐⭐ GOOD"
                    grade_color = "orange"
                elif combined >= 60:
                    grade = "⭐⭐ FAIR"
                    grade_color = "orange"
                else:
                    grade = "⭐ NEEDS WORK"
                    grade_color = "red"
                
                st.markdown(f"### :{grade_color}[{grade}]")
                
                # Detailed breakdown
                with st.expander("📊 Detailed Breakdown", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Joint Scores:**")
                        for joint, score in form_scores.items():
                            st.progress(score/100, text=f"{joint.capitalize()}: {score:.0f}/100")
                    
                    with col2:
                        st.write("**Symmetry Scores:**")
                        for joint, score in symmetry_scores.items():
                            st.progress(score/100, text=f"{joint.capitalize()}: {score:.0f}/100")
                
                # Feedback
                st.subheader("💬 Feedback & Tips")
                for tip in feedback:
                    st.write(tip)
                
                # Angle details
                with st.expander("🔢 Average Angles"):
                    cols = st.columns(2)
                    with cols[0]:
                        st.write("**Left Side:**")
                        st.write(f"Shoulder: {avg_angles['left_shoulder']:.1f}° (ideal: 180°)")
                        st.write(f"Elbow: {avg_angles['left_elbow']:.1f}° (ideal: 180°)")
                        st.write(f"Hip: {avg_angles['left_hip']:.1f}° (ideal: 180°)")
                        st.write(f"Knee: {avg_angles['left_knee']:.1f}° (ideal: 180°)")
                    with cols[1]:
                        st.write("**Right Side:**")
                        st.write(f"Shoulder: {avg_angles['right_shoulder']:.1f}° (ideal: 180°)")
                        st.write(f"Elbow: {avg_angles['right_elbow']:.1f}° (ideal: 180°)")
                        st.write(f"Hip: {avg_angles['right_hip']:.1f}° (ideal: 180°)")
                        st.write(f"Knee: {avg_angles['right_knee']:.1f}° (ideal: 180°)")
            
            # Show final frame and download button
            if run_streamlit:
                st.subheader("📹 Video Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'last_frame' in locals():
                        st.image(last_frame, channels="BGR", use_container_width=True, 
                                caption="Sample frame with pose analysis")
                
                with col2:
                    # Offer video download
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                        st.metric("Output video size", f"{file_size:.1f} MB")
                        
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 Download Analyzed Video",
                                data=f,
                                file_name=f"handstand_analysis_{int(time.time())}.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                        
                        st.info("💡 Download the video to watch it with all analyzed frames!")
                    else:
                        st.error("Output video not created")
            
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
            result = process_frame(
                image, pose, mp_pose, mp_drawing,
                drawing_spec, drawing_spec_points, rotate, return_angles=True
            )
            final_frame, angles = result
            
            if run_streamlit:
                # Calculate scores
                if angles:
                    total_score, form_scores = calculate_handstand_score(angles)
                    symmetry_score, symmetry_scores = calculate_symmetry_score(angles)
                    feedback = generate_feedback(angles, form_scores, symmetry_scores)
                    
                    # Display scores
                    st.subheader("🏆 Handstand Analysis")
                    
                    # Overall score
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Form Score", f"{total_score:.1f}/100")
                    with col2:
                        st.metric("Symmetry Score", f"{symmetry_score:.1f}/100")
                    with col3:
                        # combined = (total_score * 0.7 + symmetry_score * 0.3)
                        combined = (total_score * COMBINED_FACTOR + symmetry_score * (1-COMBINED_FACTOR))
                    
                        st.metric("Overall Score", f"{combined:.1f}/100")
                    
                    # Grade
                    if combined >= 90:
                        grade = "⭐⭐⭐⭐⭐ EXCELLENT"
                        grade_color = "green"
                    elif combined >= 80:
                        grade = "⭐⭐⭐⭐ GREAT"
                        grade_color = "blue"
                    elif combined >= 70:
                        grade = "⭐⭐⭐ GOOD"
                        grade_color = "orange"
                    elif combined >= 60:
                        grade = "⭐⭐ FAIR"
                        grade_color = "orange"
                    else:
                        grade = "⭐ NEEDS WORK"
                        grade_color = "red"
                    
                    st.markdown(f"### :{grade_color}[{grade}]")
                    
                    # Detailed breakdown
                    with st.expander("📊 Detailed Breakdown", expanded=True):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Joint Scores:**")
                            for joint, score in form_scores.items():
                                st.progress(score/100, text=f"{joint.capitalize()}: {score:.0f}/100")
                        
                        with col2:
                            st.write("**Symmetry Scores:**")
                            for joint, score in symmetry_scores.items():
                                st.progress(score/100, text=f"{joint.capitalize()}: {score:.0f}/100")
                    
                    # Feedback
                    st.subheader("💬 Feedback & Tips")
                    for tip in feedback:
                        st.write(tip)
                    
                    # Angle details
                    with st.expander("🔢 Measured Angles"):
                        cols = st.columns(3)
                        with cols[0]:
                            st.write("**Left Side:**")
                            st.write(f"Shoulder: {angles['left_shoulder']}° (ideal: 180°)")
                            st.write(f"Elbow: {angles['left_elbow']}° (ideal: 180°)")
                            st.write(f"Hip: {angles['left_hip']}° (ideal: 180°)")
                            st.write(f"Knee: {angles['left_knee']}° (ideal: 180°)")
                        with cols[1]:
                            st.write("**Right Side:**")
                            st.write(f"Shoulder: {angles['right_shoulder']}° (ideal: 180°)")
                            st.write(f"Elbow: {angles['right_elbow']}° (ideal: 180°)")
                            st.write(f"Hip: {angles['right_hip']}° (ideal: 180°)")
                            st.write(f"Knee: {angles['right_knee']}° (ideal: 180°)")
                        with cols[2]:
                            st.write("**Weight factor**")
                            st.write(f"Shoulder: {WEIGHTS['shoulder']}")
                            st.write(f"Elbow: {WEIGHTS['elbow']}")
                            st.write(f"Hip: {WEIGHTS['hip']}")
                            st.write(f"Knee: {WEIGHTS['knee']}")
                
                # Show image
                st.subheader("📸 Analyzed Image")
                stframe.image(final_frame, channels="BGR", use_container_width=True)
                
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
        st.write("**Cloud Run Edition** - version 131225e")
        
        # Show Cloud Run tips
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **This app analyzes your handstand form:**
            1. Upload a video (MP4) or image (JPG)
            2. AI detects your body pose
            3. Calculates joint angles (shoulders, elbows, hips, knees)
            4. Download the analyzed video with annotations
            
            **Cloud Run Optimizations:**
            - ✅ Videos limited to 300 frames (~10 sec at 30fps)
            - ✅ Processing every 2nd frame
            - ✅ Reduced resolution for memory efficiency
            - 💡 **Download the output video** to see all frames smoothly!
            """)

        detection_confidence = st.sidebar.number_input("Detection confidence", 0.0, 1.0, 0.5) 
        tracking_confidence = st.sidebar.number_input("Tracking confidence", 0.0, 1.0, 0.5) 
        rotate = st.sidebar.checkbox("Rotate 180°", False)
        
        # Processing mode
        processing_mode = st.sidebar.radio(
            "Processing mode",
            ["🎬 Live Preview (slower)", "⚡ Batch Process (faster)"],
            index=1
        )
        show_live = processing_mode.startswith("🎬")
        
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
        show_live = locals().get('show_live', True)
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
            detection_confidence, tracking_confidence, 0, rotate, 
            show_live if run_streamlit else True)
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