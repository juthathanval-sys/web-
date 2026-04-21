import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import time

# --- 1. นิยามโครงสร้าง SimpleCNN ---
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
    model_map = {
        "DenseNet121": ("best_densenet121.pth", models.densenet121),
        "ResNet18": ("best_resnet18.pth", models.resnet18),
        "SimpleCNN": ("best_simplecnn.pth", SimpleCNN)
    }
    
    checkpoint, model_fn = model_map[model_name]
    
    if os.path.exists(checkpoint):
        if model_name == "DenseNet121":
            model = model_fn(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, 3)
        elif model_name == "ResNet18":
            model = model_fn(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 3)
        else: # SimpleCNN
            model = model_fn(num_classes=3)
            
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.eval()
        return model, True
    return None, False

# --- 3. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Shrimp Health Dashboard", layout="wide", page_icon="🦐")

st.markdown("""
    <style>
    .stProgress .st-bp { background-color: #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

# แถบด้านข้าง
st.sidebar.title("⚙️ Settings")
selected_name = st.sidebar.selectbox("Model Selection:", ("DenseNet121", "ResNet18", "SimpleCNN"))

if st.sidebar.button("Clear All Images", key="clear_all"):
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.rerun()

st.title("🦐 Shrimp Diseases Classification")
st.write(f"Analyzing with model: **{selected_name}**")

model, model_ready = load_selected_model(selected_name)
classes = ["Normal", "EHP", "AHPND"]
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 4. ส่วนอัปโหลด (พับได้) ---
with st.expander("อัปโหลดรูปภาพ", expanded=True):
    uploader_key = str(st.session_state.get("uploader_key", 0))
    uploaded_files = st.file_uploader(
        "เลือกรูปภาพกุ้ง (JPG/PNG)", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True, 
        key=uploader_key
    )

if not model_ready:
    st.error(f"❌ ไม่พบไฟล์โมเดล 'best_{selected_name.lower()}.pth' ในระบบ")
    st.info(f"ไฟล์ที่พบ: {os.listdir('.')}")

elif uploaded_files:
    # เพิ่ม key ป้องกัน Duplicate ID
    if st.button("เริ่มการ Classification", use_container_width=True, key="run_btn"):
        st.divider()
        results_list = []
        
        # แสดงสถานะการรัน
        with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
            start_time = time.time()

            # --- 5. ส่วนแสดงผลรายรูป ---
            with st.expander("ผลการวิเคราะห์รายรูป", expanded=True):
                for file in uploaded_files:
                    img = Image.open(file).convert("RGB")
                    img_tensor = tfms(img).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(img_tensor)
                        prob = torch.softmax(output, dim=1)[0]
                        idx = torch.argmax(prob).item()
                    
                    res_class = classes[idx]
                    conf = prob[idx].item()

                    with st.container(border=True):
                        col_img, col_info = st.columns([1, 2])
                        with col_img:
                            st.image(img, use_container_width=True)
                        with col_info:
                            color = "green" if res_class == "Normal" else "red"
                            st.markdown(f"**ไฟล์:** `{file.name}`")
                            st.markdown(f"### ผลวินิจฉัย: :{color}[{res_class}]")
                            st.write(f"ความมั่นใจ: {conf*100:.2f}%")
                            st.progress(conf)
                    
                    results_list.append({
                        "Filename": file.name, 
                        "Result": res_class, 
                        "Confidence": f"{conf*100:.2f}%",
                        "Status": "Healthy" if res_class == "Normal" else "Infected"
                    })
            
            duration = time.time() - start_time

        # แจ้งเตือนเมื่อเสร็จ (อยู่นอก Spinner)
        st.success(f"✅ วิเคราะห์เสร็จสิ้น! {len(results_list)} รูป (เวลา {duration:.2f} วินาที)")
        st.toast("Analysis Complete!", icon="🦐")

        # --- 6. ส่วนสรุปสถิติ ---
        with st.expander("📊 ขั้นตอนที่ 3: สรุปสถิติการวิเคราะห์", expanded=True):
            df = pd.DataFrame(results_list)
            total = len(results_list)
            normal = sum(1 for x in results_list if x["Result"] == "Normal")
            disease = total - normal
            
            c1, c2, c3 = st.columns(3)
            c1.metric("จำนวนรูปทั้งหมด", f"{total} รูป")
            c2.metric("กุ้งปกติ", f"{normal} รูป")
            c3.metric("ตรวจพบโรค", f"{disease} รูป", delta=disease if disease > 0 else None, delta_color="inverse")
            
            if total > 0:
                st.write("**สัดส่วนผลการวินิจฉัย:**")
                st.bar_chart(df["Result"].value_counts())

        # --- 7. ตารางข้อมูลและดาวน์โหลด ---
        with st.expander("📋 ขั้นตอนที่ 4: ตารางข้อมูลและดาวน์โหลดรายงาน", expanded=False):
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 ดาวน์โหลดรายงาน (CSV)",
                data=csv,
                file_name="shrimp_report.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_btn" # ใส่ key ป้องกัน Duplicate ID
            )