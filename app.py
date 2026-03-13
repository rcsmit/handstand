# Handstand Analyzer
# Detects wrist, elbow, shoulder, hip, knee angles
# Calculates corrective torques (Nm) and gives an improvement score
#
# URL: https://handstandanalyzer-481925201210.europe-west4.run.app/

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import os
import shutil
from pathlib import Path
import gc

# ── Environment ──────────────────────────────────────────────────────────────
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["GLOG_minloglevel"] = "2"
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'

# ── Constants ─────────────────────────────────────────────────────────────────

# Body-segment mass fractions (Winter's Biomechanics Tables)
SEGMENT_MASS = {
    'hand':      0.006,
    'forearm':   0.016,
    'upper_arm': 0.028,
    'trunk_head': 0.578,   # trunk 0.497 + head 0.081
    'thigh':     0.100,
    'shank':     0.0465,
    'foot':      0.0145,
}

# Ideal angles for a perfect handstand (all 180° = fully straight)
IDEAL = {
    'wrist':    180,
    'elbow':    180,
    'shoulder': 180,
    'hip':      180,
    'knee':     180,
}

# Scoring weights (must sum to 1.0)
WEIGHTS = {
    'wrist':    0.10,
    'elbow':    0.25,
    'shoulder': 0.30,
    'hip':      0.25,
    'knee':     0.10,
}

RESIZE_FACTOR = 0.80   # resize input image before processing


# ── MediaPipe setup ───────────────────────────────────────────────────────────

@st.cache_resource
def setup_mediapipe():
    """Copy MediaPipe models to /tmp (required on Cloud Run)."""
    try:
        os.makedirs('/tmp', exist_ok=True)
        models_src = Path(mp.__file__).parent / "modules"
        models_dst = Path('/tmp') / 'mediapipe_modules'
        if models_src.exists() and not models_dst.exists():
            shutil.copytree(models_src, models_dst, dirs_exist_ok=True)
            for f in models_dst.rglob('*'):
                if f.is_file():
                    os.chmod(str(f), 0o666)
        os.environ['MEDIAPIPE_MODEL_PATH'] = str(models_dst)
    except Exception as e:
        st.warning(f"Model setup warning: {e}")
    return mp.solutions.pose, mp.solutions.drawing_utils


# ── Geometry helpers ──────────────────────────────────────────────────────────

def calculate_angle(a, b, c) -> float:
    """Interior angle at vertex b, formed by segments b→a and b→c (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _lm_xy(landmarks, mp_pose, name):
    p = landmarks[mp_pose.PoseLandmark[name].value]
    return np.array([p.x, p.y])


# ── Torque / momentum calculation ─────────────────────────────────────────────

def estimate_scale_m_per_unit(landmarks, mp_pose, body_height_m: float) -> float:
    """
    Estimate conversion factor (metres per normalised unit).
    Uses the shoulder-to-ankle distance as reference (~75 % of body height).
    """
    shoulder = _lm_xy(landmarks, mp_pose, 'LEFT_SHOULDER')
    ankle    = _lm_xy(landmarks, mp_pose, 'LEFT_ANKLE')
    dist_norm = float(np.linalg.norm(shoulder - ankle))
    if dist_norm < 0.01:
        return body_height_m          # fallback
    shoulder_to_ankle_m = body_height_m * 0.753
    return shoulder_to_ankle_m / dist_norm


def calculate_torques(landmarks, mp_pose,
                      body_weight_kg: float = 70.0,
                      body_height_m: float = 1.75) -> dict:
    """
    Estimate corrective torque (Nm) at each key joint.

    In a handstand the hands are the base of support.
    The horizontal (x-axis) displacement of each segment's centre of mass
    from the joint creates the moment that the muscles must counteract.

        T_joint = Σ (m_i · g · |Δx_i|)

    where the sum runs over all segments *distal* to the joint
    (i.e. further from the hands — further from the floor in a handstand).

    NOTE: MediaPipe x increases left→right and y increases top→bottom.
    In a handstand the body is inverted, so 'above' in the body tree
    corresponds to higher y values (lower in the image).
    """
    g = 9.81  # m/s²
    scale = estimate_scale_m_per_unit(landmarks, mp_pose, body_height_m)

    def pt(name):
        return _lm_xy(landmarks, mp_pose, name) * scale  # → metres

    wrist    = pt('LEFT_WRIST')
    elbow    = pt('LEFT_ELBOW')
    shoulder = pt('LEFT_SHOULDER')
    hip      = pt('LEFT_HIP')
    knee     = pt('LEFT_KNEE')
    ankle    = pt('LEFT_ANKLE')
    foot     = pt('LEFT_FOOT_INDEX')

    M = body_weight_kg  # total mass

    # Segment CoM positions (midpoint rule)
    com = {
        'hand':       (wrist    + elbow)    / 2,   # simplified: wrist≈hand
        'forearm':    (wrist    + elbow)    / 2,
        'upper_arm':  (elbow    + shoulder) / 2,
        'trunk_head': (shoulder + hip)      / 2,
        'thigh':      (hip      + knee)     / 2,
        'shank':      (knee     + ankle)    / 2,
        'foot':       (ankle    + foot)     / 2,
    }

    def torque_at(joint_pos, segment_list):
        """Sum of m·g·|dx| for listed segments."""
        return sum(
            SEGMENT_MASS[seg] * M * g * abs(com[seg][0] - joint_pos[0])
            for seg in segment_list
        )

    # Segments *above* each joint (moving away from hands):
    above_wrist    = ['forearm', 'upper_arm', 'trunk_head', 'thigh', 'shank', 'foot']
    above_elbow    = ['upper_arm', 'trunk_head', 'thigh', 'shank', 'foot']
    above_shoulder = ['trunk_head', 'thigh', 'shank', 'foot']
    above_hip      = ['thigh', 'shank', 'foot']
    above_knee     = ['shank', 'foot']

    return {
        'wrist':    round(torque_at(wrist,    above_wrist),    2),
        'elbow':    round(torque_at(elbow,    above_elbow),    2),
        'shoulder': round(torque_at(shoulder, above_shoulder), 2),
        'hip':      round(torque_at(hip,      above_hip),      2),
        'knee':     round(torque_at(knee,     above_knee),     2),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def joint_score(angle: float, ideal: float) -> float:
    """0–100; loses 2 points per degree of deviation."""
    return max(0.0, 100.0 - abs(angle - ideal) * 2.0)


def overall_score(angles: dict):
    """Returns (total_score, per-joint scores dict)."""
    scores = {j: joint_score(angles.get(f'left_{j}', 0), IDEAL[j]) for j in WEIGHTS}
    total  = sum(scores[j] * WEIGHTS[j] for j in scores)
    return total, scores


def get_grade(score: float):
    thresholds = [
        (92, "⭐⭐⭐⭐⭐ ELITE",      "green"),
        (82, "⭐⭐⭐⭐ EXCELLENT",    "green"),
        (70, "⭐⭐⭐ GOOD",           "blue"),
        (55, "⭐⭐ FAIR",             "orange"),
        ( 0, "⭐ NEEDS WORK",        "red"),
    ]
    for threshold, label, color in thresholds:
        if score >= threshold:
            return label, color
    return "⭐ NEEDS WORK", "red"


# ── Feedback / improvement tips ───────────────────────────────────────────────

def generate_tips(angles: dict, torques: dict) -> list:
    """
    Returns list of (joint, issue_title, detailed_advice) tuples.
    """
    tips = []

    wrist = angles.get('left_wrist', 180)
    if wrist < 155:
        tips.append(("Wrist", "significant wrist deviation",
            "Your wrist is substantially bent. Build wrist mobility with daily wrist circles, "
            "wrist push-ups and prayer-stretch holds. Also check hand placement width — "
            "shoulder-width apart is optimal."))
    elif wrist < 172:
        tips.append(("Wrist", "slight wrist deviation",
            "Push evenly through all four knuckles to align the wrist. "
            "Micro-adjust finger pressure: more pressure on the index finger corrects forward lean."))

    elbow = angles.get('left_elbow', 180)
    if elbow < 160:
        tips.append(("Elbow", "bent elbow — SAFETY RISK",
            "Bent elbows dramatically increase injury risk and waste energy. "
            "Build straight-arm strength with wall handstands, pike push-ups and "
            "Jefferson curls. Never train freestanding with consistently bent elbows."))
    elif elbow < 175:
        tips.append(("Elbow", "slightly bent elbow",
            "Focus on 'locking out' the elbows by actively contracting your triceps. "
            "Practice wall handstand holds (30 s) with conscious elbow extension cues."))

    shoulder = angles.get('left_shoulder', 180)
    if shoulder < 160:
        tips.append(("Shoulder", "closed shoulder — major alignment issue",
            "Your shoulders are not fully elevated. This is the most common handstand fault. "
            "Drill shoulder flexibility: chest openers, wall slides, and overhead band distractions. "
            "Practice 'shrug to ears' wall handstand: push the floor away and reach as tall as possible."))
    elif shoulder < 175:
        tips.append(("Shoulder", "partially closed shoulder",
            "Actively push up through your shoulders (protract and elevate scapulae). "
            "Cue: 'push the ground away from you'. Superset with active shoulder flexibility work."))

    hip = angles.get('left_hip', 180)
    if hip < 155:
        tips.append(("Hip", "strong banana / arched back",
            "This significantly shifts your centre of mass and wastes energy. "
            "Core strength is the fix: hollow body holds (3×30 s daily), dead-bug series, "
            "and anti-extension plank work. Also check hip flexor tightness — stretch hip flexors daily."))
    elif hip < 173:
        tips.append(("Hip", "slight hip flexion / arch",
            "Engage your core: brace as if about to take a punch. "
            "Posterior pelvic tilt cue: 'tuck your tailbone slightly'. "
            "Add hollow body rocks to your warm-up."))

    knee = angles.get('left_knee', 180)
    if knee < 165:
        tips.append(("Knee", "bent knees",
            "Extend your legs fully and point your toes. "
            "Squeeze your quadriceps hard. Bent knees indicate quad weakness or body-awareness gaps. "
            "Practice wall handstands focusing solely on keeping legs glued together and locked."))
    elif knee < 177:
        tips.append(("Knee", "slightly bent knees",
            "Lock out your knees: imagine squeezing a piece of paper between your thighs. "
            "Point your toes — it naturally helps knee extension."))

    max_t = max(torques.values()) if torques else 0
    if max_t > 25:
        tips.append(("Balance", f"very high corrective torque ({max_t:.1f} Nm)",
            "Your body segments are significantly displaced from the vertical stacking line. "
            "Use a wall to find the feeling of perfect vertical alignment, "
            "then gradually move away. Film yourself to spot the misalignment."))
    elif max_t > 12:
        tips.append(("Balance", f"moderate corrective torque ({max_t:.1f} Nm)",
            "Some misalignment is present. Work on body-awareness drills: "
            "kick-up to wall and slowly peel off one vertebra at a time."))

    if not tips:
        tips.append(("Overall", "outstanding form!",
            "Your handstand looks excellent. Focus on extending hold time, "
            "refining finger balance control, and exploring advanced variations "
            "(straddle, one-arm progressions, pirouette)."))

    return tips


# ── Image processing ──────────────────────────────────────────────────────────

def process_image(image, pose, mp_pose, mp_drawing, rotate: bool, text_color_name: str):
    """
    Returns (annotated_bgr_image, angles_dict, landmarks) or (image, None, None).
    """
    text_color = (0, 0, 0) if text_color_name == "black" else (255, 255, 255)
    bg_color   = (255, 255, 255) if text_color_name == "black" else (0, 0, 0)

    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w * RESIZE_FACTOR), int(h * RESIZE_FACTOR)))
    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw = rgb.shape[:2]

    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    if results.pose_landmarks is None:
        out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(out, "No pose detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return out, None, None

    lm = results.pose_landmarks.landmark

    def get(name):
        p = lm[mp_pose.PoseLandmark[name].value]
        return [p.x, p.y]

    # Key landmarks
    l_idx   = get('LEFT_INDEX');   r_idx   = get('RIGHT_INDEX')
    l_wrist = get('LEFT_WRIST');   r_wrist = get('RIGHT_WRIST')
    l_elbow = get('LEFT_ELBOW');   r_elbow = get('RIGHT_ELBOW')
    l_shldr = get('LEFT_SHOULDER');r_shldr = get('RIGHT_SHOULDER')
    l_hip   = get('LEFT_HIP');     r_hip   = get('RIGHT_HIP')
    l_knee  = get('LEFT_KNEE');    r_knee  = get('RIGHT_KNEE')
    l_ankle = get('LEFT_ANKLE');   r_ankle = get('RIGHT_ANKLE')
    l_foot  = get('LEFT_FOOT_INDEX'); r_foot = get('RIGHT_FOOT_INDEX')

    # ── Angle calculations ──
    angles = {
        'left_wrist':    int(calculate_angle(l_idx,   l_wrist, l_elbow)),
        'right_wrist':   int(calculate_angle(r_idx,   r_wrist, r_elbow)),
        'left_elbow':    int(calculate_angle(l_shldr, l_elbow, l_wrist)),
        'right_elbow':   int(calculate_angle(r_shldr, r_elbow, r_wrist)),
        'left_shoulder': int(calculate_angle(l_hip,   l_shldr, l_elbow)),
        'right_shoulder':int(calculate_angle(r_hip,   r_shldr, r_elbow)),
        'left_hip':      int(calculate_angle(l_shldr, l_hip,   l_knee)),
        'right_hip':     int(calculate_angle(r_shldr, r_hip,   r_knee)),
        'left_knee':     int(calculate_angle(l_hip,   l_knee,  l_ankle)),
        'right_knee':    int(calculate_angle(r_hip,   r_knee,  r_ankle)),
    }

    # ── Hide face landmarks ──
    HIDE = ['LEFT_EYE', 'RIGHT_EYE', 'LEFT_EYE_INNER', 'RIGHT_EYE_INNER',
            'LEFT_EYE_OUTER', 'RIGHT_EYE_OUTER', 'NOSE',
            'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_EAR', 'RIGHT_EAR',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HEEL', 'RIGHT_HEEL']
    for name in HIDE:
        lm[mp_pose.PoseLandmark[name].value].visibility = 0

    # ── Draw skeleton ──
    WHITE  = (255, 255, 255)
    LBLUE  = (180, 200, 255)
    YELLOW = (255, 220, 0)

    def pt(p):
        return (int(p[0] * iw), int(p[1] * ih))

    skeleton_L = [(l_idx, l_wrist), (l_wrist, l_elbow), (l_elbow, l_shldr),
                  (l_shldr, l_hip), (l_hip, l_knee), (l_knee, l_ankle),
                  (l_ankle, l_foot)]
    skeleton_R = [(r_idx, r_wrist), (r_wrist, r_elbow), (r_elbow, r_shldr),
                  (r_shldr, r_hip), (r_hip, r_knee), (r_knee, r_ankle),
                  (r_ankle, r_foot)]

    for p1, p2 in skeleton_L:
        cv2.line(rgb, pt(p1), pt(p2), WHITE, 2)
    for p1, p2 in skeleton_R:
        cv2.line(rgb, pt(p1), pt(p2), LBLUE, 1)

    # Joint dots on left side
    for p in [l_wrist, l_elbow, l_shldr, l_hip, l_knee, l_ankle]:
        cv2.circle(rgb, pt(p), 5, YELLOW, -1)

    # ── Angle labels ──
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fsize = 0.42
    label_positions = [
        (f"wrist {angles['left_wrist']}°",    l_wrist),
        (f"elbow {angles['left_elbow']}°",    l_elbow),
        (f"shoulder {angles['left_shoulder']}°", l_shldr),
        (f"hip {angles['left_hip']}°",         l_hip),
        (f"knee {angles['left_knee']}°",       l_knee),
    ]
    for text, pos in label_positions:
        px = int(pos[0] * iw)
        py = int(pos[1] * ih)
        (tw, th), _ = cv2.getTextSize(text, font, fsize, 1)
        # semi-transparent background box
        cv2.rectangle(rgb, (px - 2, py - th - 3), (px + tw + 2, py + 3), bg_color, -1)
        cv2.putText(rgb, text, (px, py), font, fsize, text_color, 1, cv2.LINE_AA)

    out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(out_bgr, results.pose_landmarks)

    return out_bgr, angles, lm


# ── Analysis display ──────────────────────────────────────────────────────────

def show_analysis(angles: dict, landmarks, mp_pose,
                  body_weight_kg: float, body_height_m: float):
    joints = ['wrist', 'elbow', 'shoulder', 'hip', 'knee']

    total, j_scores = overall_score(angles)
    torques = calculate_torques(landmarks, mp_pose, body_weight_kg, body_height_m)
    tips    = generate_tips(angles, torques)
    grade_label, grade_color = get_grade(total)

    # ── Grade ──
    st.markdown(f"## :{grade_color}[{grade_label}]")

    # ── Top metrics ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall Score", f"{total:.1f} / 100")
    with c2:
        max_torque = max(torques.values())
        st.metric("Highest Joint Torque", f"{max_torque:.1f} Nm")
    with c3:
        worst = min(j_scores, key=j_scores.get)
        st.metric("Weakest Point", worst.capitalize())

    st.divider()

    # ── Joint angles ──
    st.subheader("📐 Joint Angles  (ideal = 180°)")
    cols = st.columns(len(joints))
    for i, j in enumerate(joints):
        angle = angles.get(f'left_{j}', 0)
        delta = angle - IDEAL[j]
        with cols[i]:
            # delta_color: "normal" = green for positive, "inverse" flips that
            st.metric(j.capitalize(), f"{angle}°",
                      delta=f"{delta:+.0f}°",
                      delta_color="normal" if abs(delta) <= 3 else "inverse")

    st.divider()

    # ── Joint scores ──
    st.subheader("📊 Joint Scores")
    for j in joints:
        s = j_scores[j]
        icon = "🟢" if s >= 80 else ("🟡" if s >= 55 else "🔴")
        st.progress(s / 100,
                    text=f"{icon} {j.capitalize()}: **{s:.0f}/100**  "
                         f"(weight {WEIGHTS[j]:.0%})")

    st.divider()

    # ── Torque table ──
    st.subheader("⚙️ Corrective Torques (Nm)")
    st.caption(
        "The rotational force each joint must overcome to hold the current position. "
        "A perfectly vertical alignment gives 0 Nm. Higher torque = larger misalignment."
    )
    cols = st.columns(len(joints))
    for i, j in enumerate(joints):
        t = torques[j]
        icon = "🟢" if t < 5 else ("🟡" if t < 15 else "🔴")
        with cols[i]:
            st.metric(f"{icon} {j.capitalize()}", f"{t:.1f} Nm")

    st.divider()

    # ── Improvement tips ──
    st.subheader("💡 How to Improve")
    for joint_name, issue_title, advice in tips:
        with st.expander(f"**{joint_name}** — {issue_title}"):
            st.write(advice)

    gc.collect()


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Handstand Analyzer",
        page_icon="🤸",
        layout="wide",
    )
    st.title("🤸 Handstand Analyzer")
    st.caption("Upload a handstand photo · get instant angle analysis, torque scores, and coaching tips.")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")
        body_weight = st.number_input("Body weight (kg)",  30.0, 200.0, 70.0, 1.0)
        body_height = st.number_input("Body height (m)",   1.0,  2.50,  1.75, 0.01)
        rotate      = st.checkbox("Rotate image 180°", value=False)
        text_color  = st.selectbox("Annotation colour", ["black", "white"])
        det_conf    = st.slider("Detection confidence", 0.1, 1.0, 0.5, 0.05)

        st.divider()
        st.markdown("""
**Ideal angles** (all 180°)

| Joint    | Weight |
|----------|--------|
| Shoulder | 30 %   |
| Elbow    | 25 %   |
| Hip      | 25 %   |
| Wrist    | 10 %   |
| Knee     | 10 %   |
""")

    # ── Upload ──
    uploaded = st.file_uploader(
        "Upload a handstand photo (side view works best)",
        type=['jpg', 'jpeg', 'png'],
    )

    if uploaded is None:
        st.info("👆 Upload a photo to start analysis.")
        with st.expander("ℹ️ How it works"):
            st.markdown("""
1. **Upload** a side-view handstand photo
2. **MediaPipe** locates 33 body landmarks
3. **Angles** are measured at wrist, elbow, shoulder, hip and knee
4. **Score** is calculated based on deviation from the ideal 180°
5. **Torques (Nm)** estimate the corrective force at each joint, using your body weight and standard biomechanical segment-mass fractions
6. **Tips** give you targeted drills to improve each weak point
""")
        st.stop()

    # ── Save temp file ──
    suffix = uploaded.name.rsplit('.', 1)[-1].lower()
    temp_path = f'/tmp/hs_upload_{int(__import__("time").time())}.{suffix}'
    with open(temp_path, 'wb') as fh:
        fh.write(uploaded.read())

    mp_pose, mp_drawing = setup_mediapipe()

    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=det_conf,
        ) as pose:

            image = cv2.imread(temp_path)
            if image is None:
                st.error("Could not read image file. Please try a different photo.")
                st.stop()

            with st.spinner("Analysing pose…"):
                img_out, angles, landmarks = process_image(
                    image, pose, mp_pose, mp_drawing, rotate, text_color
                )

        col_img, col_analysis = st.columns([1, 1], gap="large")

        with col_img:
            st.subheader("📸 Annotated Photo")
            st.image(img_out, channels="BGR", use_container_width=True)
            _, buf = cv2.imencode('.jpg', img_out)
            st.download_button(
                "📥 Download annotated image",
                data=buf.tobytes(),
                file_name="handstand_analysis.jpg",
                mime="image/jpeg",
            )

        with col_analysis:
            if angles is not None and landmarks is not None:
                show_analysis(angles, landmarks, mp_pose, body_weight, body_height)
            else:
                st.error(
                    "❌ No pose detected in this photo.  \n"
                    "Tips: use a clear side-view photo, good lighting, "
                    "and make sure your full body is visible."
                )

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
        gc.collect()


if __name__ == '__main__':
    main()
