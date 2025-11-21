import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from transformers import SamModel, SamProcessor
from controlnet_aux import OpenposeDetector
from gradio_client import Client
import torch
import os
import time

# --- COMPATIBILITY FIXES ---
try:
    from gradio_client import handle_file
except ImportError:
    from gradio_client import file as handle_file

def safe_image(st_obj, image, caption, width_param=True):
    try:
        st_obj.image(image, caption=caption, use_container_width=width_param)
    except TypeError:
        st_obj.image(image, caption=caption, use_column_width=width_param)

# Fix MediaPipe/OpenCV Conflict on macOS
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Virtual Try On",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #ffffff; color: #000000; }
    h1, h2, h3 { font-family: sans-serif; font-weight: 700; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #000000;
        color: #ffffff;
        font-weight: 600;
        border: none;
        padding: 12px;
    }
    .stButton>button:hover { background-color: #333333; color: #ffffff;}
    div.stStatus { border-radius: 10px; padding: 15px; border: 1px solid #ddd; }
    img { border-radius: 8px; border: 1px solid #eee; }
    </style>
""", unsafe_allow_html=True)

# --- 1. LOAD LOCAL AI MODELS ---
@st.cache_resource
def load_local_tools():
    # Force CPU for analysis to prevent Mac MPS float64 crash
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam_model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50").to(device)
    sam_processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    
    return sam_model, sam_processor, openpose, device

# --- 2. HELPER FUNCTIONS ---
def get_intersection_point(image_np):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    results = pose.process(image_np)
    vis_img = image_np.copy()
    point = None
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        h, w, _ = image_np.shape
        p1 = (int(lm[11].x * w), int(lm[11].y * h)) 
        p2 = (int(lm[12].x * w), int(lm[12].y * h)) 
        p3 = (int(lm[23].x * w), int(lm[23].y * h)) 
        p4 = (int(lm[24].x * w), int(lm[24].y * h)) 
        
        def intersect(a, b, c, d):
            det = (a[0]-b[0])*(c[1]-d[1]) - (a[1]-b[1])*(c[0]-d[0])
            if det == 0: return None
            x = ((a[0]*b[1] - a[1]*b[0])*(c[0]-d[0]) - (a[0]-b[0])*(c[0]*d[1] - c[1]*d[0])) / det
            y = ((a[0]*b[1] - a[1]*b[0])*(c[1]-d[1]) - (a[1]-b[1])*(c[0]*d[1] - c[1]*d[0])) / det
            return [int(x), int(y)]
            
        point = intersect(p1, p4, p2, p3)
        cv2.line(vis_img, p1, p4, (255, 215, 0), 3) # Gold lines
        cv2.line(vis_img, p2, p3, (255, 215, 0), 3)
        if point: 
            cv2.circle(vis_img, tuple(point), 10, (0, 255, 0), -1) # Green dot
            cv2.circle(vis_img, tuple(point), 12, (255, 255, 255), 2) # White border
            
    return point, vis_img

def generate_mask(image_pil, point, model, processor, device):
    if not point: point = [image_pil.width//2, image_pil.height//2]
    inputs = processor(image_pil, input_points=[[point]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    return Image.fromarray((masks[0][0][2].cpu().numpy() * 255).astype(np.uint8))

def try_on_request(client, p_path, c_path, desc, steps, seed):
    return client.predict(
        dict={"background": handle_file(p_path), "layers": [], "composite": None},
        garm_img=handle_file(c_path),
        garment_des=desc,
        is_checked=True,
        is_checked_crop=False,
        denoise_steps=steps,
        seed=seed,
        api_name="/tryon"
    )

# --- 3. UI LAYOUT ---
st.title("Virtual Try On")
st.markdown("### Professional AI Fitting Room")

with st.sidebar:
    st.header("Configuration")
    person_file = st.file_uploader("Upload Person", type=["jpg", "png", "jpeg"], key="p")
    cloth_file = st.file_uploader("Upload Garment", type=["jpg", "png", "jpeg"], key="c")
    desc = st.text_input("Garment Description", "A cool t-shirt")
    with st.expander("Advanced Settings"):
        denoise_steps = st.slider("Denoise Steps", 20, 50, 30)
        seed = st.number_input("Random Seed", value=42)

if person_file and cloth_file:
    c1, c2 = st.columns(2)
    # Save temp files
    with open("temp_person.jpg", "wb") as f: f.write(person_file.getbuffer())
    with open("temp_cloth.png", "wb") as f: f.write(cloth_file.getbuffer())
    
    p_img = Image.open("temp_person.jpg").convert("RGB")
    c_img = Image.open("temp_cloth.png").convert("RGB")
    
    safe_image(c1, p_img, "Model Input")
    safe_image(c2, c_img, "Garment Input")

    if st.button("✨ Generate Virtual Try-On", type="primary"):
        status = st.status("🚀 Initializing AI Pipeline...", expanded=True)
        try:
            # --- PHASE 1: LOCAL VISUALIZATION ---
            status.write("🧠 Analying Body Geometry & Pose...")
            sam_model, sam_processor, openpose, device = load_local_tools()
            
            # 1. Geometry (Intersection)
            p_cv2 = np.array(p_img)
            intersection, geo_viz = get_intersection_point(p_cv2)
            
            # 2. Segmentation (Mask)
            mask = generate_mask(p_img, intersection, sam_model, sam_processor, device)
            
            # 3. Pose (Skeleton)
            pose = openpose(p_img)
            
            # Display Analysis Layers Nicely
            st.divider()
            st.markdown("### 🔍 Analysis Layers")
            k1, k2, k3 = st.columns(3)
            safe_image(k1, geo_viz, "Geometric Intersection")
            safe_image(k2, mask, "Segmentation Mask")
            safe_image(k3, pose, "OpenPose Skeleton")
            
            # --- PHASE 2: CLOUD SYNTHESIS ---
            status.write("☁️ Connecting to Synthesis Engine...")
            
            try:
                client = Client("yisol/IDM-VTON")
                status.write("🎨 Generating High-Fidelity Image...")
                result = try_on_request(client, "temp_person.jpg", "temp_cloth.png", desc, denoise_steps, seed)
                
            except Exception as e:
                if "429" in str(e) or "Queue" in str(e):
                    status.write("⚠️ Primary Engine Busy. Retrying in 5s...")
                    time.sleep(5)
                    result = try_on_request(client, "temp_person.jpg", "temp_cloth.png", desc, denoise_steps, seed)
                else:
                    raise e
            
            output_path = result[0]
            status.update(label="✅ Fitting Complete!", state="complete", expanded=False)
            
            # Final Display
            st.divider()
            st.markdown("### 🎉 Final Result")
            final_col, _ = st.columns([1, 1])
            safe_image(final_col, Image.open(output_path), "Virtual Try-On Output")
            
            with open(output_path, "rb") as file:
                st.download_button("Download Final Image", file, "result.png", "image/png")

        except Exception as e:
            status.update(label="❌ Error", state="error")
            if "429" in str(e):
                st.warning("🚦 High Traffic: The public demo is currently full. Please wait 1-2 minutes and try again.")
            else:
                st.error(f"System Error: {str(e)}")
else:
    st.info("👈 Upload images in the sidebar to begin.")