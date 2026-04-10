import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# --- 1. นิยามโครงสร้าง SimpleCNN (ต้องตรงกับที่เทรนไว้) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

# --- 2. ฟังก์ชันโหลดโมเดล ---
@st.cache_resource
def load_selected_model(model_name):
    if model_name == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 3)
        checkpoint = "best_densenet121.pth"
    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        checkpoint = "best_resnet18.pth"
    elif model_name == "SimpleCNN":
        model = SimpleCNN(num_classes=3)
        checkpoint = "best_simplecnn.pth"
    
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        return model, True
    return None, False

# --- 3. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Shrimp AI Analysis Dashboard", layout="wide", page_icon="🦐")

# ตกแต่ง CSS เล็กน้อยให้ Progress Bar เป็นสีแดงตามตัวอย่าง
st.markdown("""
    <style>
    .stProgress .st-bp { background-color: #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

# แถบด้านข้าง (Sidebar)
st.sidebar.title("⚙️ AI Settings")
selected_name = st.sidebar.selectbox("Model Selection:", ("DenseNet121", "ResNet18", "SimpleCNN"))

# ปุ่มล้างข้อมูลทั้งหมด
if st.sidebar.button("🗑️ Clear All Images"):
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.rerun()

st.title("🦐 Shrimp Health Analysis Dashboard")
st.write(f"Analyzing with model: **{selected_name}**")

model, model_ready = load_selected_model(selected_name)
classes = ["Normal", "EHP", "AHPND"]

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 4. ส่วนอัปโหลดและประมวลผล ---
if model_ready:
    uploader_key = str(st.session_state.get("uploader_key", 0))
    uploaded_files = st.file_uploader(
        "Upload Shrimp Images (JPG/PNG)", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True, 
        key=uploader_key
    )

    if uploaded_files:
        if st.button("🚀 Run Analysis"):
            st.divider()
            results_for_csv = []
            
            # วนลูปประมวลผลทีละรูป
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                img_tensor = tfms(img).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)[0]
                    idx = torch.argmax(prob).item()
                
                res_class = classes[idx]
                conf = prob[idx].item()

                # --- ส่วนการแสดงผลแบบ Card (รูปซ้าย ข้อมูลขวา) ---
                with st.container(border=True):
                    col_img, col_info = st.columns([1, 2]) # แบ่งสัดส่วน 1:2
                    
                    with col_img:
                        # แสดงรูปขนาดกลาง-ใหญ่ (width=280) เพื่อให้เห็นรายละเอียดชัดเจน
                        st.image(img, width=280, caption=f"Original: {file.name}")
                    
                    with col_info:
                        st.markdown(f"### 📄 Filename: {file.name}")
                        
                        # แสดงผลลัพธ์พร้อมสีเน้นย้ำ
                        color = "green" if res_class == "Normal" else "red"
                        st.markdown(f"#### Diagnosis: :{color}[{res_class}]")
                        
                        # แถบแสดงความมั่นใจ (Confidence Bar)
                        st.write(f"**Confidence:** {conf*100:.2f}%")
                        st.progress(conf)
                        
                        # ข้อมูลเพิ่มเติม (คะแนนของทุกคลาส)
                        with st.expander("Show detail probabilities"):
                            for i, c_name in enumerate(classes):
                                st.write(f"{c_name}: {prob[i]*100:.2f}%")
                
                results_for_csv.append({
                    "Filename": file.name, 
                    "Result": res_class, 
                    "Confidence": f"{conf*100:.2f}%"
                })

            # --- ส่วนท้าย: ตารางสรุปและดาวน์โหลด ---
            st.divider()
            st.subheader("📊 Summary Table")
            df = pd.DataFrame(results_for_csv)
            st.table(df) # ใช้ st.table เพื่อให้โชว์ทุกแถวแบบคงที่
            
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Download Report (CSV)", csv, "shrimp_report.csv", "text/csv")
else:
    st.error(f"Error: Model file 'best_{selected_name.lower()}.pth' not found in current directory.")