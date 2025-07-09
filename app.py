import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import csv
import os
import base64
import io
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

# page config
st.set_page_config(page_title="Mold Detection Web Panel", layout="wide")

# model loading
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load("mold_model_optimized.pt", map_location=device)
    model.eval()
    return model, device

model, device = load_model()

# transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# CSV log file
log_file = "prediction_log.csv"
def init_log_file():
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["image_name", "prediction", "confidence", "timestamp"])

init_log_file()

# --- SIDEBAR NAVIGATION ---
page = st.sidebar.selectbox("Navigation", ["Main Panel", "Messages"])

# main panel
if page == "Main Panel":
    st.title("Mold Detection Web Panel")
    st.markdown("Upload images to detect mold and explore analysis below.")

    uploaded_images = st.file_uploader("Upload image(s):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        st.sidebar.header("Settings")
        threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

        cols = st.columns(4)
        new_logs = []

        for idx, image_file in enumerate(uploaded_images):
            image = Image.open(image_file).convert("RGB")
            display_image = image.resize((250, 250))
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                confidence = output.item()
                prediction = "Mold Detected" if confidence > threshold else "Clean"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_logs.append([image_file.name, prediction, f"{confidence:.2f}", timestamp])

            color = "red" if prediction == "Mold Detected" else "green"

            # Görseli base64'e çevirme
            buffered = io.BytesIO()
            display_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            with cols[idx % 4]:
                st.markdown(f"""
                    <div style='display: flex; flex-direction: column; align-items: center; min-height: 350px;'>
                        <img src='data:image/png;base64,{img_base64}' style='height: 250px; width: 250px; object-fit: cover; border-radius: 12px;'/>
                        <div class="stMarkdown" data-testid="stMarkdown">
                            <div data-testid="stMarkdownContainer" class="st-emotion-cache-seewz2 erovr380">
                                <p><strong>Prediction:</strong> <code style="color:{color}">{prediction}</code></p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        # Save logs
        with open(log_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(new_logs)

        st.success(f"✅ {len(new_logs)} predictions saved to prediction_log.csv.")
        st.markdown("---")

        # dashboard
        st.header("📊 Prediction Summary")

        df_new = pd.DataFrame(new_logs, columns=["image_name", "prediction", "confidence", "timestamp"])
        df_new["confidence"] = pd.to_numeric(df_new["confidence"])

        pred_counts = df_new["prediction"].value_counts().rename_axis('Prediction').reset_index(name='Count')

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(pred_counts, x='Prediction', y='Count',
                             color='Prediction', title="Bar Chart",
                             color_discrete_map={'Clean': '#00C851', 'Mold Detected': '#ff4444'})
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(pred_counts, names='Prediction', values='Count',
                             title="Pie Chart", color='Prediction',
                             color_discrete_map={'Clean': '#00C851', 'Mold Detected': '#ff4444'})
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0])
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Confidence Distribution")
        fig_hist = px.histogram(df_new, x='confidence', nbins=20,
                                title="Model Confidence Histogram",
                                color_discrete_sequence=['#007BFF'])
        st.plotly_chart(fig_hist, use_container_width=True)

# message panel
elif page == "Messages":
    st.title("📩 Summary Reports")

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["prediction"] = df["prediction"].replace({
            "Temiz": "Clean",
            "Küf Tespit Edildi": "Mold Detected"
        })

        today = datetime.now().date()
        last_7_days = today - timedelta(days=7)

        report_type = st.radio("Choose Report Type:", ["Daily", "Weekly"], horizontal=True)

        if report_type == "Daily":
            filtered = df[df["timestamp"].dt.date == today]
            st.subheader(f"🗓️ Daily Report: {today}")
        else:
            filtered = df[df["timestamp"].dt.date >= last_7_days]
            st.subheader(f"Weekly Report: {last_7_days} to {today}")

        if not filtered.empty:
            st.markdown(f"*Total Predictions:* {len(filtered)}")
            st.markdown(f"*Clean:* {(filtered['prediction'] == 'Clean').sum()}")
            st.markdown(f"*Mold Detected:* {(filtered['prediction'] == 'Mold Detected').sum()}")

            fig = px.histogram(filtered, x="prediction", color="prediction",
                               title="Prediction Overview",
                               color_discrete_map={'Clean': '#00C851', 'Mold Detected': '#ff4444'})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Raw Report Table")
            st.dataframe(filtered[::-1])
        else:
            st.warning("No predictions available for this period.")
    else:
        st.warning("Prediction log not found.")
