import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroSense | Precision Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics, glassmorphism, responsive cards, and vibrant colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }

    /* Top Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(27, 67, 50, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        color: #f1faee;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        max-width: 850px;
        line-height: 1.5;
        color: #d8f3dc;
    }

    .badge-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 0.35rem 0.85rem;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(45, 106, 79, 0.12);
    }

    .card-title {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
    }

    .card-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #1b4332;
        margin-top: 0.2rem;
    }

    /* Result Showcase Card */
    .result-box {
        background: linear-gradient(145deg, #f8fdf9 0%, #e8f5e9 100%);
        border: 2px solid #52b788;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(45, 106, 79, 0.15);
        margin: 1.5rem 0;
    }

    .crop-name-highlight {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1b4332;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0.5rem 0;
    }

    .crop-badge {
        background-color: #2d6a4f;
        color: white;
        padding: 0.4rem 1.2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1rem;
        display: inline-block;
    }

    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1.5rem;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# RESOURCE & MODEL LOADING
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EDA_DIR = os.path.join(BASE_DIR, "eda_plots")

@st.cache_resource
def load_models_and_metadata():
    # Load metadata
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
    # Load models
    rf_path = os.path.join(MODELS_DIR, "crop_recommendation_rf.pkl")
    dt_path = os.path.join(MODELS_DIR, "crop_recommendation_dt.pkl")
    root_path = os.path.join(BASE_DIR, "crop_recommendation_model.pkl")

    rf_model = None
    dt_model = None

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
    elif os.path.exists(root_path):
        rf_model = joblib.load(root_path)

    if os.path.exists(dt_path):
        dt_model = joblib.load(dt_path)

    return rf_model, dt_model, metadata

@st.cache_data
def load_dataset():
    data_paths = [
        os.path.join(BASE_DIR, "Crop_recommendation.csv"),
        os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
    ]
    for p in data_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

rf_model, dt_model, metadata = load_models_and_metadata()
df_data = load_dataset()

# -------------------------------------------------------------
# CROP ICONS AND METADATA MAPPING
# -------------------------------------------------------------
CROP_ICONS = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🧆", "kidneybeans": "🫘",
    "pigeonpeas": "🌱", "mothbeans": "🌿", "mungbean": "🥗", "blackgram": "🌰",
    "lentil": "🍲", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍏",
    "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "👕",
    "jute": "🧵", "coffee": "☕"
}

# -------------------------------------------------------------
# SIDEBAR NAVIGATION & CONFIGURATION
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### 🌾 AgroSense Controls")
    
    # Model Selection
    model_choice = st.selectbox(
        "Select Machine Learning Engine:",
        ["Random Forest Classifier (Ensemble - 99.55% Acc)", "Decision Tree Classifier (Gini - 97.95% Acc)"],
        index=0
    )
    
    active_model = rf_model if "Random Forest" in model_choice else dt_model
    if active_model is None:
        active_model = rf_model or dt_model

    st.markdown("---")
    st.markdown("#### ⚡ Quick Agro-Scenario Presets")
    st.caption("Load verified test presets to evaluate real-world crop environments:")
    
    preset = st.selectbox(
        "Choose Preset Environment:",
        [
            "Custom User Input",
            "Rice Field (Monsoon, High Rain & N)",
            "Apple Orchard (Temperate, High K)",
            "Cotton Belt (Black Soil, Warm)",
            "Coffee Plantation (High Altitude, High Humidity)",
            "Chickpea / Pulses (Semi-Arid, Low Rain)",
            "Grapes Vineyard (High P & K Demand)",
            "Watermelon (Warm Summer, Moderate Rain)"
        ]
    )

    preset_values = {
        "Custom User Input": (90.0, 42.0, 43.0, 20.88, 82.00, 6.50, 202.94),
        "Rice Field (Monsoon, High Rain & N)": (90.0, 42.0, 43.0, 21.0, 82.0, 6.5, 230.0),
        "Apple Orchard (Temperate, High K)": (20.0, 134.0, 198.0, 22.5, 92.0, 6.0, 110.0),
        "Cotton Belt (Black Soil, Warm)": (118.0, 46.0, 19.0, 24.0, 80.0, 6.8, 80.0),
        "Coffee Plantation (High Altitude, High Humidity)": (102.0, 29.0, 30.0, 25.5, 58.0, 6.7, 160.0),
        "Chickpea / Pulses (Semi-Arid, Low Rain)": (40.0, 68.0, 80.0, 19.0, 16.5, 7.3, 75.0),
        "Grapes Vineyard (High P & K Demand)": (22.0, 130.0, 200.0, 23.5, 81.5, 6.2, 70.0),
        "Watermelon (Warm Summer, Moderate Rain)": (99.0, 17.0, 50.0, 25.5, 88.0, 6.5, 50.0)
    }

    default_n, default_p, default_k, default_temp, default_hum, default_ph, default_rain = preset_values[preset]

    st.markdown("---")
    st.markdown("""
    **Project Metadata:**
    - **Group**: Group 15
    - **Dataset**: 2,200 Balanced Records
    - **Classes**: 22 Agricultural Crops
    - **Evaluation**: 5-Fold Stratified CV
    """)

# -------------------------------------------------------------
# MAIN APP HERO BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
        <span class="badge-pill">🏛️ Group 15 Mini Project</span>
        <span class="badge-pill">🌱 Precision Agriculture</span>
        <span class="badge-pill">🎯 99.55% Benchmark Accuracy</span>
        <span class="badge-pill">⚡ 5-Fold Stratified CV</span>
    </div>
    <div class="hero-title">AgroSense Precision Crop Recommender</div>
    <div class="hero-subtitle">
        An intelligent decision-support system that diagnoses soil chemistry and atmospheric conditions to recommend the highest-yielding crop for sustainable agricultural productivity.
    </div>
</div>
""", unsafe_allow_html=True)

# Top Key Performance Indicator Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="metric-card">
        <div class="card-title">Ensemble Accuracy</div>
        <div class="card-value">99.55%</div>
        <small style="color: #2d6a4f;">Random Forest Classifier</small>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="card-title">Supported Crops</div>
        <div class="card-value">22 Species</div>
        <small style="color: #2d6a4f;">Cereals, Pulses, Fruits, Cash</small>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="card-title">Input Dimensions</div>
        <div class="card-value">7 Features</div>
        <small style="color: #2d6a4f;">N-P-K & Climatic Factors</small>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="card-title">Decision Tree Acc</div>
        <div class="card-value">97.95%</div>
        <small style="color: #2d6a4f;">Gini Impurity Split</small>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------------
# TABS SETUP
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Single Recommendation",
    "📂 Batch CSV Prediction",
    "📊 Exploratory Data Analysis",
    "🔬 Model Benchmarking"
])

# -------------------------------------------------------------
# TAB 1: SINGLE PREDICTION & SOIL HEALTH
# -------------------------------------------------------------
with tab1:
    st.subheader("🌾 Soil & Agro-Climatic Parameter Diagnostic")
    st.caption("Adjust the sliders or input exact laboratory soil testing and meteorology figures below:")
    
    col_soil, col_climate = st.columns(2)
    
    with col_soil:
        st.markdown("#### 🧪 Soil Macronutrients (NPK) & Acidity")
        n_input = st.slider("Nitrogen (N) Content (mg/kg)", min_value=0.0, max_value=140.0, value=float(default_n), step=1.0, help="Nitrogen promotes lush leaf growth and chlorophyll formation.")
        p_input = st.slider("Phosphorus (P) Content (mg/kg)", min_value=5.0, max_value=145.0, value=float(default_p), step=1.0, help="Phosphorus drives healthy root systems and flower/fruit set.")
        k_input = st.slider("Potassium (K) Content (mg/kg)", min_value=5.0, max_value=205.0, value=float(default_k), step=1.0, help="Potassium regulates internal water balance and disease resilience.")
        ph_input = st.slider("Soil pH Value (Acidity / Alkalinity)", min_value=3.5, max_value=10.0, value=float(default_ph), step=0.1, help="Optimal nutrient uptake happens between pH 6.0 and 7.5.")
        
        # Real-time Soil pH diagnostic badge
        if ph_input < 5.5:
            st.warning(f"⚠️ Soil is **Strongly Acidic** (pH {ph_input:.1f}). Lime treatment may be required for sensitive crops.")
        elif ph_input > 8.0:
            st.warning(f"⚠️ Soil is **Alkaline / Calcareous** (pH {ph_input:.1f}). Gypsum or organic matter addition recommended.")
        else:
            st.success(f"✓ Soil pH is in the **Ideal Agronomic Range** (pH {ph_input:.1f}).")

    with col_climate:
        st.markdown("#### 🌦️ Meteorology & Environmental Conditions")
        temp_input = st.slider("Ambient Temperature (°C)", min_value=8.0, max_value=45.0, value=float(default_temp), step=0.5, help="Average temperature during growing season.")
        hum_input = st.slider("Relative Atmospheric Humidity (%)", min_value=14.0, max_value=100.0, value=float(default_hum), step=1.0, help="Percentage moisture content in ambient air.")
        rain_input = st.slider("Annual Precipitation / Rainfall (mm)", min_value=20.0, max_value=300.0, value=float(default_rain), step=1.0, help="Seasonal or annual precipitation available for crop canopy.")
        
        if rain_input < 50.0:
            st.info("💧 Arid / Low rainfall zone: Favors drought-resilient legumes and semi-arid pulses.")
        elif rain_input > 200.0:
            st.info("🌧️ Heavy precipitation zone: Favors water-intensive crops like Rice and Jute.")

    st.write("")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        recommend_btn = st.button("🌿 Generate Precision Crop Recommendation", type="primary", use_container_width=True)

    if recommend_btn:
        input_data = pd.DataFrame([{
            'N': n_input,
            'P': p_input,
            'K': k_input,
            'temperature': temp_input,
            'humidity': hum_input,
            'ph': ph_input,
            'rainfall': rain_input
        }])

        try:
            # Predict crop
            prediction = active_model.predict(input_data)[0]
            
            # Predict probabilities if supported
            probabilities = None
            if hasattr(active_model, "predict_proba"):
                probabilities = active_model.predict_proba(input_data)[0]
                class_names = active_model.classes_
                top_indices = np.argsort(probabilities)[::-1][:3]
                top_crops = [(class_names[i], probabilities[i]) for i in top_indices]
            else:
                top_crops = [(prediction, 1.0)]

            st.balloons()

            icon = CROP_ICONS.get(prediction.lower(), "🌱")
            best_prob = top_crops[0][1] * 100

            st.markdown(f"""
            <div class="result-box">
                <div class="crop-badge">TOP MATCH FOUND WITH {best_prob:.1f}% CONFIDENCE</div>
                <div class="crop-name-highlight">{icon} {prediction.upper()}</div>
                <p style="font-size: 1.1rem; color: #2d6a4f; margin: 0;">
                    Optimal cultivation candidate based on current NPK and agro-climatic conditions.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Top 3 Alternative Candidates
            st.markdown("#### 🏅 Top-3 Recommended Agronomic Matches")
            prob_cols = st.columns(3)
            for idx, (c_name, c_prob) in enumerate(top_crops):
                with prob_cols[idx]:
                    st.metric(
                        label=f"Rank #{idx+1}: {CROP_ICONS.get(c_name.lower(), '🌱')} {c_name.capitalize()}",
                        value=f"{c_prob*100:.1f}%",
                        delta="Recommended" if idx == 0 else "Alternative Option"
                    )
                    st.progress(float(c_prob))

            # Cultivation Guide Card
            st.markdown("---")
            st.markdown(f"### 📖 Agronomic Cultivation Dossier: **{prediction.capitalize()}**")
            
            profiles = metadata.get("crop_profiles", {})
            crop_info = profiles.get(prediction.lower(), {
                "season": "Seasonal", "water": "Moderate", "soil": "Fertile loam", "ph": "6.0 - 7.5",
                "desc": "Cultivation should follow local agricultural university package of practices."
            })

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.info(f"**🗓️ Optimal Season**\n\n{crop_info.get('season')}")
            with d2:
                st.info(f"**💧 Water Requirement**\n\n{crop_info.get('water')}")
            with d3:
                st.info(f"**🌍 Preferred Soil**\n\n{crop_info.get('soil')}")
            with d4:
                st.info(f"**⚗️ Tolerant pH Range**\n\n{crop_info.get('ph')}")

            st.markdown(f"**Agricultural Summary:** {crop_info.get('desc')}")

            # Soil Nutrient Health Advisor
            st.markdown("---")
            st.markdown("#### 🔬 Soil Nutrient Diagnostic & Fertilization Advisor")
            s1, s2, s3 = st.columns(3)
            with s1:
                if n_input < 30:
                    st.warning("⚠️ **Nitrogen is Deficient** (< 30 mg/kg). Consider applying Urea or organic compost before sowing.")
                elif n_input > 100:
                    st.info("ℹ️ **Nitrogen is Plentiful** (> 100 mg/kg). Avoid excessive synthetic nitrogen to prevent lodging.")
                else:
                    st.success("✓ **Nitrogen is Balanced** (30 - 100 mg/kg).")

            with s2:
                if p_input < 25:
                    st.warning("⚠️ **Phosphorus is Deficient** (< 25 mg/kg). Apply Diammonium Phosphate (DAP) or rock phosphate.")
                elif p_input > 90:
                    st.info("ℹ️ **Phosphorus is High** (> 90 mg/kg). Favorable for high-energy root/tuber development.")
                else:
                    st.success("✓ **Phosphorus is Balanced** (25 - 90 mg/kg).")

            with s3:
                if k_input < 25:
                    st.warning("⚠️ **Potassium is Deficient** (< 25 mg/kg). Apply Muriate of Potash (MOP) to boost pest immunity.")
                elif k_input > 120:
                    st.info("ℹ️ **Potassium is High** (> 120 mg/kg). Exceptional for fruit sugar accumulation (Grapes/Apples).")
                else:
                    st.success("✓ **Potassium is Balanced** (25 - 120 mg/kg).")

        except Exception as err:
            st.error(f"Error during recommendation inference: {err}")

# -------------------------------------------------------------
# TAB 2: BATCH CSV PREDICTOR
# -------------------------------------------------------------
with tab2:
    st.subheader("📂 Batch Processing & Multi-Plot Farm Diagnosis")
    st.write("Upload a CSV file containing multiple soil and environmental test results to receive batch crop recommendations simultaneously.")

    # Download Template Button
    sample_df = pd.DataFrame([
        {"N": 90, "P": 42, "K": 43, "temperature": 20.88, "humidity": 82.0, "ph": 6.5, "rainfall": 202.94},
        {"N": 20, "P": 134, "K": 198, "temperature": 22.5, "humidity": 92.0, "ph": 6.0, "rainfall": 110.0},
        {"N": 118, "P": 46, "K": 19, "temperature": 24.0, "humidity": 80.0, "ph": 6.8, "rainfall": 80.0},
        {"N": 40, "P": 68, "K": 80, "temperature": 19.0, "humidity": 16.5, "ph": 7.3, "rainfall": 75.0}
    ])
    csv_template = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Input CSV Template",
        data=csv_template,
        file_name="crop_recommendation_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload Test Soil Data (CSV)", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            req_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            
            missing_cols = [c for c in req_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}. Please use the provided template.")
            else:
                st.success(f"✓ Successfully loaded {len(batch_df)} samples!")
                
                preds = active_model.predict(batch_df[req_cols])
                batch_df["Recommended_Crop"] = preds
                
                if hasattr(active_model, "predict_proba"):
                    probs = np.max(active_model.predict_proba(batch_df[req_cols]), axis=1) * 100
                    batch_df["Confidence (%)"] = np.round(probs, 2)
                    
                st.dataframe(batch_df, use_container_width=True)
                
                # Distribution of batch recommendations
                st.markdown("#### 📊 Recommended Crop Distribution for Uploaded Samples")
                crop_counts = batch_df["Recommended_Crop"].value_counts().reset_index()
                crop_counts.columns = ["Crop", "Count"]
                st.bar_chart(data=crop_counts, x="Crop", y="Count", color="#2d6a4f")
                
                # Download result
                out_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📤 Download Batch Recommendations CSV",
                    data=out_csv,
                    file_name="crop_recommendations_output.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")

# -------------------------------------------------------------
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
with tab3:
    st.subheader("📊 Exploratory Data Analysis & Agronomic Patterns")
    st.write("Visualizations generated from 2,200 agricultural observations across 22 crop classes:")

    eda_selection = st.selectbox(
        "Select EDA Visualization:",
        [
            "Target Class Distribution (Uniform Balance)",
            "Nutrient & Environmental Feature Distributions",
            "Pearson Correlation Matrix",
            "Outlier Spread Across Features (Boxplots)",
            "Average N-P-K Macronutrient Demand by Crop",
            "Climatic Niches: Rainfall vs. Temperature"
        ]
    )

    plot_mapping = {
        "Target Class Distribution (Uniform Balance)": "01_crop_distribution.png",
        "Nutrient & Environmental Feature Distributions": "02_feature_distributions.png",
        "Pearson Correlation Matrix": "03_correlation_heatmap.png",
        "Outlier Spread Across Features (Boxplots)": "04_outlier_boxplots.png",
        "Average N-P-K Macronutrient Demand by Crop": "05_npk_ratio_by_crop.png",
        "Climatic Niches: Rainfall vs. Temperature": "06_rainfall_vs_temp.png"
    }

    img_file = plot_mapping.get(eda_selection)
    img_path = os.path.join(EDA_DIR, img_file) if img_file else None

    if img_path and os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.warning("Chart image not found. Please verify `train_and_evaluate.py` has generated the plot.")

    # Interactive crop comparator
    if df_data is not None:
        st.markdown("---")
        st.subheader("⚖️ Interactive Crop Nutrient Comparator")
        st.caption("Compare the average nutritional and environmental demands of two distinct crops side-by-side:")
        
        all_crops = sorted(df_data['label'].unique())
        c1, c2 = st.columns(2)
        with c1:
            crop_a = st.selectbox("Select Crop A:", all_crops, index=all_crops.index("rice"))
        with c2:
            crop_b = st.selectbox("Select Crop B:", all_crops, index=all_crops.index("coffee"))

        comp_df = df_data.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean().loc[[crop_a, crop_b]].T
        comp_df.columns = [crop_a.capitalize(), crop_b.capitalize()]
        comp_df['Difference'] = comp_df[crop_a.capitalize()] - comp_df[crop_b.capitalize()]
        st.dataframe(comp_df.round(2), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: MODEL BENCHMARKING & EVALUATION
# -------------------------------------------------------------
with tab4:
    st.subheader("🔬 Machine Learning Model Benchmarking & Performance Comparison")
    st.write("Quantitative comparison of 5 machine learning classifiers evaluated on the stratified 440-sample test set:")

    comp_csv_path = os.path.join(BASE_DIR, "model_comparison.csv")
    if os.path.exists(comp_csv_path):
        comp_df = pd.read_csv(comp_csv_path)
        st.dataframe(comp_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score (Weighted)', '5-Fold CV Mean'], color='#d8f3dc'), use_container_width=True)
    
    st.write("")
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("#### 🏆 Benchmark Metrics Comparison")
        comp_chart = os.path.join(EDA_DIR, "10_model_comparison_chart.png")
        if os.path.exists(comp_chart):
            st.image(comp_chart, use_container_width=True)
    with bc2:
        st.markdown("#### 🌲 Feature Importance (RF vs. DT)")
        feat_chart = os.path.join(EDA_DIR, "07_feature_importance.png")
        if os.path.exists(feat_chart):
            st.image(feat_chart, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Confusion Matrix Inspection")
    cm_choice = st.radio("Select Confusion Matrix to View:", ["Random Forest (99.55% Accuracy)", "Decision Tree (97.95% Accuracy)"], horizontal=True)
    if "Random Forest" in cm_choice:
        rf_cm_path = os.path.join(EDA_DIR, "09_random_forest_confusion_matrix.png")
        if os.path.exists(rf_cm_path):
            st.image(rf_cm_path, use_container_width=True)
    else:
        dt_cm_path = os.path.join(EDA_DIR, "08_decision_tree_confusion_matrix.png")
        if os.path.exists(dt_cm_path):
            st.image(dt_cm_path, use_container_width=True)



# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.9rem; padding: 1rem;'>"
    "🌱 <strong>AgroSense: Precision Crop Recommendation System</strong> | Machine Learning Fundamentals Mini Project (Group 15)<br>"
    "Developed with Streamlit, Scikit-Learn, and Python 3.9."
    "</div>",
    unsafe_allow_html=True
)
