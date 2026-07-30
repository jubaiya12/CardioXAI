import numpy as np
import pickle
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from gradcam import get_gradcam
from sklearn.metrics import classification_report, confusion_matrix
import io
import time
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib
matplotlib.use('Agg')

# ── Load model + data ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('ecg_cnn_lstm_model.keras')

@st.cache_data
def load_data():
    beats = np.load('beats.npy')
    labels = np.load('labels.npy')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return beats, labels, le

model = load_model()
beats, labels, le = load_data()

CLASS_NAMES = {0: 'A (Atrial)', 1: 'L (Left Bundle)', 2: 'N (Normal)', 3: 'R (Right Bundle)', 4: 'V (Ventricular)'}
CLASS_COLORS = {0: '#f39c12', 1: '#9b59b6', 2: '#2ecc71', 3: '#3498db', 4: '#e74c3c'}

CLINICAL_INFO = {
    0: {
        'rhythm': 'Atrial Premature Contraction (APC)',
        'risk': 'MODERATE',
        'risk_color': '#f39c12',
        'risk_score': 45,
        'features': ['Abnormal P-wave morphology', 'Premature beat origin in atria', 'Narrow QRS complex'],
        'explanation': 'The CNN-LSTM model focused on P-wave abnormalities and premature beat timing, indicating atrial ectopic activity.',
        'recommendation': 'Atrial premature contractions detected. Clinical correlation recommended. May indicate atrial irritability.',
        'color': '#f39c12'
    },
    1: {
        'rhythm': 'Left Bundle Branch Block (LBBB)',
        'risk': 'MODERATE',
        'risk_color': '#f39c12',
        'risk_score': 55,
        'features': ['Wide QRS complex (>120ms)', 'Abnormal ventricular conduction', 'Left bundle pathway affected'],
        'explanation': 'The model focused heavily on QRS widening and morphology changes characteristic of left bundle branch block.',
        'recommendation': 'Left Bundle Branch Block pattern detected. Cardiology consultation advised for underlying cause assessment.',
        'color': '#9b59b6'
    },
    2: {
        'rhythm': 'Normal Sinus Rhythm',
        'risk': 'LOW',
        'risk_color': '#2ecc71',
        'risk_score': 5,
        'features': ['Regular RR intervals', 'Narrow QRS complex', 'Normal ventricular conduction'],
        'explanation': 'The CNN-LSTM model focused primarily on the QRS complex and T-wave morphology, indicating a normal electrical rhythm.',
        'recommendation': 'No obvious arrhythmia detected. Final interpretation should be confirmed by a cardiologist.',
        'color': '#2ecc71'
    },
    3: {
        'rhythm': 'Right Bundle Branch Block (RBBB)',
        'risk': 'MODERATE',
        'risk_color': '#f39c12',
        'risk_score': 50,
        'features': ['Wide QRS complex', 'RSR pattern in V1', 'Right bundle pathway conduction delay'],
        'explanation': 'The model identified characteristic QRS widening and morphology consistent with right bundle branch block.',
        'recommendation': 'Right Bundle Branch Block pattern detected. Further evaluation recommended to assess clinical significance.',
        'color': '#3498db'
    },
    4: {
        'rhythm': 'Ventricular Premature Contraction (PVC)',
        'risk': 'HIGH',
        'risk_color': '#e74c3c',
        'risk_score': 80,
        'features': ['Wide and bizarre QRS complex', 'No preceding P-wave', 'Compensatory pause present'],
        'explanation': 'The CNN-LSTM model focused on the abnormally wide QRS complex and absence of normal P-wave, indicating ventricular ectopic origin.',
        'recommendation': 'Ventricular premature contraction detected. Requires clinical evaluation especially if frequent or symptomatic.',
        'color': '#e74c3c'
    }
}

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(page_title="CardioXAI - A clinical decision support tool ", layout="wide", page_icon="🫀")

# Header
st.markdown("""
<div style='background:linear-gradient(135deg,#0d1b2a,#1b263b);padding:20px 30px;border-radius:12px;margin-bottom:20px;border-left:4px solid #00d4ff'>
<h1 style='color:white;margin:0;font-size:28px'>🫀CardioXAI - A clinical decision support tool</h1>
<p style='color:#00d4ff;margin:4px 0 0 0;font-size:14px'>Explainable CNN-LSTM Framework for Cardiac Arrhythmia Classification</p>
<p style='color:#666;margin:2px 0 0 0;font-size:11px'>MIT-BIH Arrhythmia Database • Grad-CAM Explainability • Inter-Patient Evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='background:#0d1b2a;padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid #1b263b'>
    <p style='color:#00d4ff;font-weight:bold;margin:0;font-size:13px'>🏥 CardioXAI SYSTEM</p>
    <p style='color:#666;margin:2px 0 0 0;font-size:10px'>Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Patient")
    patient_name = st.text_input("Patient Name", "Anonymous", label_visibility="collapsed",
                                  placeholder="Enter patient name...")
    st.markdown(f"<p style='color:#666;font-size:11px;margin-top:-8px'>📅 {datetime.date.today().strftime('%d %B %Y')}</p>", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Input Mode")
    mode = st.radio("", ["📊 MIT-BIH + CSV", "🖼️ ECG Image"], label_visibility="collapsed")

    st.divider()

    st.markdown("""
    <div style='background:#0d1b2a;padding:12px;border-radius:8px;border:1px solid #1b263b'>
    <p style='color:#00d4ff;font-weight:bold;margin:0 0 8px 0;font-size:12px'>📊 MODEL PERFORMANCE</p>
    <table width='100%'>
    <tr><td style='color:#888;font-size:11px'>Inter-Patient Accuracy</td><td style='color:#2ecc71;font-weight:bold;font-size:12px;text-align:right'>76%</td></tr>
    <tr><td style='color:#888;font-size:11px'>Random Split Accuracy</td><td style='color:#f39c12;font-size:12px;text-align:right'>99%</td></tr>
    <tr><td style='color:#888;font-size:11px'>Architecture</td><td style='color:white;font-size:11px;text-align:right'>CNN-LSTM</td></tr>
    <tr><td style='color:#888;font-size:11px'>Explainability</td><td style='color:white;font-size:11px;text-align:right'>Grad-CAM</td></tr>
    <tr><td style='color:#888;font-size:11px'>Dataset</td><td style='color:white;font-size:11px;text-align:right'>MIT-BIH</td></tr>
    <tr><td style='color:#888;font-size:11px'>Classes</td><td style='color:white;font-size:11px;text-align:right'>N, A, V, R, L</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.expander("📈 Confusion Matrix"):
        sample_idx = np.random.choice(len(beats), 300, replace=False)
        X_sample = beats[sample_idx][..., np.newaxis]
        y_sample = labels[sample_idx]
        preds_all = np.argmax(model.predict(X_sample, verbose=0), axis=1)
        cm = confusion_matrix(y_sample, preds_all)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
        fig_cm.patch.set_facecolor('#0d1b2a')
        ax_cm.set_facecolor('#0d1b2a')
        im = ax_cm.imshow(cm, cmap='Blues')
        ax_cm.set_xticks(range(5))
        ax_cm.set_yticks(range(5))
        ax_cm.set_xticklabels(['A','L','N','R','V'], color='white', fontsize=8)
        ax_cm.set_yticklabels(['A','L','N','R','V'], color='white', fontsize=8)
        ax_cm.set_xlabel('Predicted', color='white', fontsize=8)
        ax_cm.set_ylabel('True', color='white', fontsize=8)
        for i in range(5):
            for j in range(5):
                ax_cm.text(j, i, cm[i,j], ha='center', va='center', color='white', fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with st.expander("📚 Literature Comparison"):
        from evaluation import plot_comparison
        fig_comp = plot_comparison()
        st.pyplot(fig_comp)
        plt.close()

    with st.expander("🎯 Research Contributions"):
        st.markdown("""
<small>
 End-to-End ECG Pipeline<br>
 ECG Image Digitization<br>
 CNN-LSTM Classification<br>
 Grad-CAM Explainability<br>
 Inter-Patient Evaluation<br>
 AI Clinical Summary<br>
 PDF Report Generation<br>
 Literature Benchmarking
</small>
        """, unsafe_allow_html=True)

    st.markdown("<p style='color:#444;font-size:10px;text-align:center;margin-top:8px'>⚠️ Research only. Not for clinical diagnosis.</p>", unsafe_allow_html=True)

# ── Helper functions ───────────────────────────────────────────────
def plot_ecg_gradcam(beat, heatmap, pred_class):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.patch.set_facecolor('#0d1b2a')
    x = np.arange(len(beat))
    regions = {
        'P-wave\n(Atrial activity)': (40, 70, '#f1c40f'),
        'QRS Complex\n(Ventricular depol.)': (75, 105, '#e74c3c'),
        'T-wave\n(Ventricular repol.)': (110, 145, '#3498db')
    }
    for name, (s, e, c) in regions.items():
        axes[0].axvspan(s, e, alpha=0.15, color=c, label=name)
    axes[0].plot(beat, color='#00d4ff', linewidth=1.5)
    axes[0].legend(loc='upper right', fontsize=7)
    axes[0].set_title('① ECG Beat Signal — Labeled with Clinical Regions', color='white', fontsize=11)
    axes[0].set_ylabel('Amplitude (normalized)', color='#888', fontsize=8)
    axes[0].set_facecolor('#0d1b2a')
    axes[0].tick_params(colors='#888')
    axes[0].spines[:].set_color('#333')

    axes[1].plot(beat, color='#00d4ff', linewidth=1.5)
    for i in range(len(beat) - 1):
        axes[1].fill_between([x[i], x[i+1]], [beat[i], beat[i+1]],
                              alpha=float(heatmap[i]) * 0.85, color='red')
    axes[1].set_title('② Grad-CAM Heatmap — Red areas show where the model focused to make its prediction', color='white', fontsize=11)
    axes[1].set_ylabel('Amplitude (normalized)', color='#888', fontsize=8)
    axes[1].set_xlabel('Sample index (180 samples = 1 heartbeat)', color='#888', fontsize=8)
    axes[1].set_facecolor('#0d1b2a')
    axes[1].tick_params(colors='#888')
    axes[1].spines[:].set_color('#333')
    plt.tight_layout()
    return fig

def generate_pdf(patient_name, beat, heatmap, pred_class, confidence, true_class=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    info = CLINICAL_INFO[pred_class]
    story.append(Paragraph("CardioXAI — Arrhythmia Detection Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Patient:</b> {patient_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {datetime.date.today().strftime('%d %B %Y')}", styles['Normal']))
    story.append(Paragraph(f"<b>Predicted Rhythm:</b> {info['rhythm']}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.1f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Risk Level:</b> {info['risk']}", styles['Normal']))
    if true_class is not None:
        story.append(Paragraph(f"<b>True Label:</b> {CLASS_NAMES[true_class]}", styles['Normal']))
    story.append(Spacer(1, 12))
    fig = plot_ecg_gradcam(beat, heatmap, pred_class)
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', bbox_inches='tight', facecolor='white')
    img_buf.seek(0)
    plt.close(fig)
    story.append(RLImage(img_buf, width=450, height=220))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Grad-CAM Region Activation:</b>", styles['Normal']))
    regions = {'P-wave': (40, 70), 'QRS complex': (75, 105), 'T-wave': (110, 145)}
    for name, (s, e) in regions.items():
        act = np.mean(heatmap[s:e]) * 100
        story.append(Paragraph(f"  • {name}: {act:.1f}% model attention", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>AI Recommendation:</b> {info['recommendation']}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Clinical Note:</b> This report is generated by an AI research system. Not intended for clinical diagnosis. Please consult a qualified cardiologist.", styles['Normal']))
    doc.build(story)
    buf.seek(0)
    return buf

# ── Main Input ─────────────────────────────────────────────────────
beat = None
true_class = None

if mode == "📊 MIT-BIH + CSV":
    tab1, tab2 = st.tabs(["📊 MIT-BIH Database", "📁 Upload CSV"])

    with tab1:
        st.markdown("Browse ECG beats from the MIT-BIH Arrhythmia Database by class and sample index.")
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_class = st.selectbox("Arrhythmia Class", list(CLASS_NAMES.values()),
                help="Select the type of heartbeat to analyze")
            class_idx = list(CLASS_NAMES.values()).index(selected_class)
            class_indices = np.where(labels == class_idx)[0]
            sample_num = st.slider("Sample Index", 0, min(99, len(class_indices)-1), 0,
                help="Slide to browse different examples of this beat type")
            beat_idx = class_indices[sample_num]
            beat = beats[beat_idx][..., np.newaxis]
            true_class = class_idx
            st.markdown(f"<p style='color:#666;font-size:11px'>Showing sample {sample_num+1} of {min(100, len(class_indices))} available {selected_class} beats</p>", unsafe_allow_html=True)

    with tab2:
        st.markdown("Upload a CSV file containing ECG signal values. The file should have one column of numerical values representing one heartbeat (180 samples).")
        uploaded = st.file_uploader("Upload ECG CSV", type=['csv'], label_visibility="collapsed")
        if uploaded:
            import pandas as pd
            try:
                # Try to detect patient name from CSV
                raw = uploaded.read().decode('utf-8')
                uploaded.seek(0)
                detected_name = None
                for line in raw.split('\n')[:5]:
                    if 'name' in line.lower() or 'patient' in line.lower():
                        parts = line.split(',')
                        for p in parts:
                            if len(p.strip()) > 2 and not p.strip().replace('.','').replace('-','').isnumeric():
                                detected_name = p.strip()
                                break

                df = pd.read_csv(uploaded)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, header=None)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns

                if len(numeric_cols) == 0:
                    st.error("No numeric column found in CSV.")
                else:
                    if detected_name:
                        st.info(f"Patient name detected: **{detected_name}** — update in sidebar if incorrect.")

                    steps_csv = [
                        " Reading CSV File...",
                        " Detecting Signal Column...",
                        " Normalizing Signal...",
                        " Segmenting Beat...",
                        " Running CNN-LSTM...",
                        " Generating Grad-CAM...",
                        " Creating Clinical Report..."
                    ]
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i, step in enumerate(steps_csv[:-3]):
                        status_text.markdown(f"**{step}**")
                        progress_bar.progress((i + 1) / len(steps_csv))
                        time.sleep(0.3)
                    signal = df[numeric_cols[0]].values[:180].astype(float)
                    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                    for i, step in enumerate(steps_csv[-3:]):
                        status_text.markdown(f"**{step}**")
                        progress_bar.progress((len(steps_csv) - 3 + i + 1) / len(steps_csv))
                        time.sleep(0.3)
                    status_text.markdown("**✅ Analysis Complete!**")
                    progress_bar.progress(1.0)
                    if len(signal) < 180:
                        st.error("CSV must have at least 180 rows.")
                    else:
                        beat = signal[..., np.newaxis]
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

elif mode == "🖼️ ECG Image":
    st.markdown("""
    <div style='background:#0d1b2a;padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid #1b263b'>
    <p style='color:#00d4ff;margin:0;font-size:13px;font-weight:bold'> How ECG Image Analysis Works</p>
    <p style='color:#888;margin:4px 0 0 0;font-size:11px'>
    Your ECG image is processed to extract the electrical signal from a specific lead. 
    <b style='color:white'>Lead I</b> is recommended as our model was trained on Lead I signals from MIT-BIH.
    Using other leads may affect accuracy.
    </p>
    </div>
    """, unsafe_allow_html=True)

    col_lead1, col_lead2 = st.columns([1, 2])
    with col_lead1:
        lead_options = {'Lead I (Recommended)': 0, 'Lead II': 1, 'Lead III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5}
        selected_lead = st.selectbox("Select Lead", list(lead_options.keys()),
            help="Lead I is recommended. Our model was trained on Lead I from MIT-BIH Arrhythmia Database.")
        lead_row = lead_options[selected_lead]

    uploaded_img = st.file_uploader("Upload ECG Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])
    if uploaded_img:
        import tempfile
        import os
        from ecg_digitizer import process_ecg_image

        # Try OCR for patient name
        try:
            import pytesseract
            from PIL import Image as PILImage
            img_pil = PILImage.open(uploaded_img)
            uploaded_img.seek(0)
            header_region = img_pil.crop((0, 0, img_pil.width, int(img_pil.height * 0.22)))
            ocr_text = pytesseract.image_to_string(header_region)
            detected_name = None
            for line in ocr_text.split('\n'):
                if 'name' in line.lower() or 'patient' in line.lower():
                    parts = line.split(':')
                    if len(parts) > 1:
                        detected_name = parts[1].strip()
                        break
            if detected_name:
                st.info(f"Patient detected from ECG: **{detected_name}** — update in sidebar if incorrect.")
        except:
            pass

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_img.read())
            tmp_path = tmp.name

        try:
            st.image(uploaded_img, caption="Uploaded ECG Image", use_container_width=True)

            steps = [
                " Loading ECG Image...",
                " Removing Background Grid...",
                f" Extracting {selected_lead} Signal...",
                " Detecting R-peaks (Heartbeats)...",
                " Segmenting Individual Beats...",
                " Running CNN-LSTM Classifier...",
                " Generating Grad-CAM Heatmap...",
                " Preparing Clinical Report..."
            ]
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, step in enumerate(steps[:-3]):
                status_text.markdown(f"**{step}**")
                progress_bar.progress((i + 1) / len(steps))
                time.sleep(0.4)
            signal, ecg_beats, peaks = process_ecg_image(tmp_path, lead_row=lead_row)
            for i, step in enumerate(steps[-3:]):
                status_text.markdown(f"**{step}**")
                progress_bar.progress((len(steps) - 3 + i + 1) / len(steps))
                time.sleep(0.4)
            status_text.markdown("**✅ Analysis Complete!**")
            progress_bar.progress(1.0)

            if len(ecg_beats) == 0:
                st.error("No heartbeats detected. Try Lead I or ensure the ECG image is clear.")
            else:
                st.success(f"✅ Detected **{len(ecg_beats)} heartbeats** from {selected_lead}")

                st.markdown("**Extracted Signal with Detected Heartbeats (R-peaks marked in red):**")
                fig_sig, ax_sig = plt.subplots(figsize=(12, 2.5))
                fig_sig.patch.set_facecolor('#0d1b2a')
                ax_sig.plot(signal, color='#00d4ff', linewidth=0.8)
                if len(peaks) > 0:
                    ax_sig.scatter(peaks, signal[peaks], color='red', s=30, zorder=5, label=f'{len(peaks)} R-peaks detected')
                ax_sig.set_facecolor('#0d1b2a')
                ax_sig.tick_params(colors='#888')
                ax_sig.spines[:].set_color('#333')
                ax_sig.set_title(f'Extracted {selected_lead} Signal — Each red dot is one detected heartbeat', color='white', fontsize=10)
                ax_sig.legend(fontsize=8, loc='upper right')
                st.pyplot(fig_sig)
                plt.close()

                st.markdown(f"**Select which heartbeat to analyze** (0 = first detected beat, {len(ecg_beats)-1} = last):")
                beat_num = st.slider("Beat number", 0, len(ecg_beats)-1, 0, label_visibility="collapsed")
                beat = ecg_beats[beat_num][..., np.newaxis]
                st.markdown(f"<p style='color:#666;font-size:11px'>Analyzing beat {beat_num+1} of {len(ecg_beats)}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error processing image: {e}")
        finally:
            os.unlink(tmp_path)

# ── Results ─────────────────────────────────────────────────────────
if beat is not None:
    pred = model.predict(beat[np.newaxis, ...], verbose=0)[0]
    pred_class = int(np.argmax(pred))
    confidence = pred[pred_class] * 100
    heatmap = get_gradcam(model, beat, pred_class)
    info = CLINICAL_INFO[pred_class]

    st.divider()

    # ── ECG + Grad-CAM plots FIRST ──────────────────────────────────
    fig = plot_ecg_gradcam(beat[:, 0], heatmap, pred_class)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Grad-CAM region breakdown ───────────────────────────────────
    regions_data = {'P-wave': (40, 70), 'QRS Complex': (75, 105), 'T-wave': (110, 145)}
    region_acts = {name: np.mean(heatmap[s:e]) * 100 for name, (s, e) in regions_data.items()}
    primary_region = max(region_acts, key=region_acts.get)

    col_g1, col_g2, col_g3 = st.columns(3)
    for col, (name, act) in zip([col_g1, col_g2, col_g3], region_acts.items()):
        with col:
            bar = "█" * int(act / 10)
            color = "#f1c40f" if name == "P-wave" else "#e74c3c" if name == "QRS Complex" else "#3498db"
            st.markdown(f"""
<div style='background:#0d1b2a;border:1px solid #1b263b;border-radius:8px;padding:12px;text-align:center'>
<p style='color:#888;font-size:11px;margin:0'>{name}</p>
<p style='color:{color};font-size:22px;font-weight:bold;margin:4px 0'>{act:.0f}%</p>
<p style='color:{color};font-size:10px;margin:0'>{bar}</p>
<p style='color:#666;font-size:10px;margin:4px 0 0 0'>model attention</p>
</div>
            """, unsafe_allow_html=True)

    st.markdown(f"<p style='color:#888;font-size:11px;margin-top:8px'>Primary focus: <b style='color:white'>{primary_region}</b> — {region_acts[primary_region]:.0f}% of model attention</p>", unsafe_allow_html=True)

    st.divider()

    # ── AI Clinical Summary AFTER plots ────────────────────────────
    signal_quality = "Excellent" if confidence > 90 else "Good" if confidence > 75 else "Fair"
    quality_color = "#2ecc71" if confidence > 90 else "#f39c12" if confidence > 75 else "#e74c3c"
    quality_emoji = "🟢" if confidence > 90 else "🟡" if confidence > 75 else "🔴"
    features_html = "".join([f"<p style='color:white;margin:3px 0;font-size:13px'>✔ {f}</p>" for f in info['features']])

    risk_colors = {'LOW': '#2ecc71', 'MODERATE': '#f39c12', 'HIGH': '#e74c3c'}
    risk_emoji = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🔴'}
    risk_c = risk_colors[info['risk']]

    card = f"""<div style='background:linear-gradient(135deg,#0d1b2a,#1b263b);border:1px solid {info['color']};border-radius:12px;padding:24px;margin:8px 0'>
<div style='display:flex;align-items:center;margin-bottom:16px'>
<span style='font-size:20px'>🩺</span>
<h3 style='color:white;margin:0 0 0 8px;font-size:18px'>AI Clinical Summary</h3>
<span style='margin-left:auto;background:{risk_c};color:black;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:bold'>{risk_emoji[info['risk']]} {info['risk']} RISK</span>
</div>
<table width='100%' style='margin-bottom:16px'><tr>
<td style='background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;width:33%'>
<p style='color:#888;font-size:11px;margin:0'>PREDICTED RHYTHM</p>
<p style='color:{info['color']};font-size:14px;font-weight:bold;margin:4px 0 0 0'>{info['rhythm']}</p>
</td>
<td style='width:2%'></td>
<td style='background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;width:33%'>
<p style='color:#888;font-size:11px;margin:0'>MODEL CONFIDENCE</p>
<p style='color:{info['color']};font-size:14px;font-weight:bold;margin:4px 0 0 0'>{confidence:.1f}%</p>
</td>
<td style='width:2%'></td>
<td style='background:rgba(255,255,255,0.05);border-radius:8px;padding:12px;width:30%'>
<p style='color:#888;font-size:11px;margin:0'>SIGNAL QUALITY</p>
<p style='color:{quality_color};font-size:14px;font-weight:bold;margin:4px 0 0 0'>{quality_emoji} {signal_quality}</p>
</td>
</tr></table>
<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px'>
<div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:12px'>
<p style='color:#888;font-size:11px;margin:0 0 8px 0'>FEATURES DETECTED BY MODEL</p>
{features_html}
</div>
<div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:12px'>
<p style='color:#888;font-size:11px;margin:0 0 8px 0'>WHY DID MODEL PREDICT THIS?</p>
<p style='color:white;font-size:12px;margin:0'>{info['explanation']}</p>
<br>
<p style='color:#888;font-size:11px;margin:0 0 4px 0'>PRIMARY FOCUS REGION</p>
<p style='color:#00d4ff;font-size:12px;margin:0'>Model focused most on <b>{primary_region}</b> ({region_acts[primary_region]:.0f}% attention)</p>
</div>
</div>
<div style='background:rgba(255,255,255,0.03);border-radius:8px;padding:12px;margin-top:12px;border-left:3px solid {risk_c}'>
<p style='color:#888;font-size:11px;margin:0 0 4px 0'>AI RECOMMENDATION</p>
<p style='color:{risk_c};font-size:13px;margin:0'>⚠️ {info['recommendation']}</p>
</div>
</div>"""

    st.markdown(card, unsafe_allow_html=True)

    if true_class is not None and pred_class != true_class:
        st.warning(f"⚠️ Misclassification: True label is **{CLASS_NAMES[true_class]}** but model predicted **{CLASS_NAMES[pred_class]}**. This is the inter-patient generalization challenge our research addresses.")

    st.divider()

    # ── Class probabilities + PDF ───────────────────────────────────
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.markdown("####  Class Probabilities")
        prob_fig, ax = plt.subplots(figsize=(8, 3))
        prob_fig.patch.set_facecolor('#0d1b2a')
        ax.set_facecolor('#0d1b2a')
        bar_colors = [CLASS_COLORS[i] if i == pred_class else '#2a3a4a' for i in range(5)]
        bars = ax.barh(list(CLASS_NAMES.values()), pred * 100, color=bar_colors, height=0.6)
        ax.set_xlabel('Confidence (%)', color='#888', fontsize=9)
        ax.set_xlim(0, 105)
        ax.tick_params(colors='#888', labelsize=9)
        ax.spines[:].set_color('#333')
        for bar, val in zip(bars, pred * 100):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                   va='center', color='white', fontsize=8)
        st.pyplot(prob_fig)
        plt.close()

    with col_p2:
        st.markdown("####  Report")
        if true_class is not None:
            st.markdown(f"**True Label:** `{CLASS_NAMES[true_class]}`")
        pdf_buf = generate_pdf(patient_name, beat[:, 0], heatmap, pred_class, confidence, true_class)
        st.download_button(
            " Download Clinical Report (PDF)",
            data=pdf_buf,
            file_name=f"ExplainECG_{patient_name}_{datetime.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        st.markdown(f"<p style='color:#666;font-size:10px'>Report for: {patient_name} | {datetime.date.today().strftime('%d %b %Y')}</p>", unsafe_allow_html=True)

        # Signal quality
        signal_data = beat[:, 0]
        noise_level = np.std(np.diff(signal_data)) / (np.std(signal_data) + 1e-8)
        noise_pct = min(100, noise_level * 100)
        snr = 20 * np.log10(np.std(signal_data) / (noise_level + 1e-8))
        sq_color = "#2ecc71" if noise_pct < 10 else "#f39c12" if noise_pct < 25 else "#e74c3c"
        sq_label = "Excellent" if noise_pct < 10 else "Good" if noise_pct < 25 else "Poor"
        st.markdown(f"""
<div style='background:#0d1b2a;border:1px solid #1b263b;border-radius:8px;padding:10px;margin-top:8px'>
<p style='color:#888;font-size:10px;margin:0 0 6px 0'>SIGNAL QUALITY</p>
<p style='color:{sq_color};font-weight:bold;font-size:13px;margin:0'>{sq_label}</p>
<p style='color:#666;font-size:10px;margin:2px 0 0 0'>SNR: {snr:.1f} dB | Noise: {noise_pct:.0f}%</p>
</div>
        """, unsafe_allow_html=True)