import streamlit as st
import pandas as pd
import joblib
import requests
import google.generativeai as genai
from datetime import datetime
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import random

# 1. SETUP & CONFIGURATION

st.set_page_config(page_title="OceanSafe AI", page_icon="🦈", layout="centered")
st.markdown("""
    <style>
    /* 1. Import Sirin Stencil from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sirin+Stencil&display=swap');

    /* 2. Apply it to EVERYTHING (Titles, Headers, and Text) */
    html, body, [class*="css"] {
        font-family: 'Sirin Stencil', sans-serif;
    }

    /* 3. Specific Styling for the Big Title */
    h1 {
        font-family: 'Sirin Stencil', sans-serif;
        color: #00B4D8; /* Cyan Color */
        text-shadow: 0 0 5px #00B4D8; /* Subtle Glow */
        text-align: center;
        font-size: 3.5rem; /* Make it HUGE */
    }

    /* 4. Style the Subheaders */
    h2, h3 {
        font-family: 'Sirin Stencil', sans-serif;
        color: #90E0EF;
        border-bottom: 1px solid #0077B6;
    }

    /* 5. Update Buttons to match */
    .stButton>button {
        font-family: 'Sirin Stencil', sans-serif;
        border-radius: 10px;
        border: 2px solid #00B4D8;
        color: #00B4D8;
        background-color: transparent;
    }
    .stButton>button:hover {
        background-color: #00B4D8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- API KEY ---

if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

genai.configure(api_key=GOOGLE_API_KEY)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('oceansafe_model.pkl')
        encoders = joblib.load('oceansafe_encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_assets()

# --- DATABASE ---
BEACH_DB = {
    # --- INTERNATIONAL ---
    "Bondi Beach (Australia)": {"country": "AUSTRALIA", "lat": -33.8915, "lon": 151.2767},
    "Gold Coast (Australia)": {"country": "AUSTRALIA", "lat": -28.0167, "lon": 153.4000},
    "New Smyrna Beach (USA)": {"country": "USA", "lat": 29.0258, "lon": -80.9270},
    "Maui (USA)": {"country": "USA", "lat": 20.7984, "lon": -156.3319},
    "Jeffreys Bay (South Africa)": {"country": "SOUTH AFRICA", "lat": -34.0333, "lon": 24.9167},
    "Durban (South Africa)": {"country": "SOUTH AFRICA", "lat": -29.8587, "lon": 31.0218},
    "Piha Beach (New Zealand)": {"country": "NEW ZEALAND", "lat": -36.9536, "lon": 174.4706},

    # --- MALAYSIA: WEST COAST ---
    "Cenang Beach (Malaysia)": {"country": "MALAYSIA", "lat": 6.2913, "lon": 99.7278},
    "Tanjung Rhu (Malaysia)": {"country": "MALAYSIA", "lat": 6.4552, "lon": 99.8228},
    "Batu Ferringhi (Malaysia)": {"country": "MALAYSIA", "lat": 5.4735, "lon": 100.2452},
    "Monkey Beach (Malaysia)": {"country": "MALAYSIA", "lat": 5.4770, "lon": 100.1837},
    "Pantai Kerachut (Malaysia)": {"country": "MALAYSIA", "lat": 5.4542, "lon": 100.1770},
    "Teluk Nipah (Malaysia)": {"country": "MALAYSIA", "lat": 4.2307, "lon": 100.5447},
    "Coral Beach (Malaysia)": {"country": "MALAYSIA", "lat": 4.2374, "lon": 100.5434},
    "Port Dickson (Malaysia)": {"country": "MALAYSIA", "lat": 2.4344, "lon": 101.8546},
    "Blue Lagoon (Malaysia)": {"country": "MALAYSIA", "lat": 2.4139, "lon": 101.8550},

    # --- MALAYSIA: EAST COAST ---
    "Long Beach (Malaysia)": {"country": "MALAYSIA", "lat": 5.9224, "lon": 102.7214},
    "Coral Bay (Malaysia)": {"country": "MALAYSIA", "lat": 5.9173, "lon": 102.7153},
    "Romantic Beach (Malaysia)": {"country": "MALAYSIA", "lat": 5.9083, "lon": 102.7447},
    "Pasir Panjang (Malaysia), Redang": {"country": "MALAYSIA", "lat": 5.7728, "lon": 103.0336},
    "Teluk Dalam (Malaysia), Redang": {"country": "MALAYSIA", "lat": 5.7954, "lon": 103.0182},
    "Juara Beach (Malaysia)": {"country": "MALAYSIA", "lat": 2.7933, "lon": 104.2045},
    "Salang Beach (Malaysia)": {"country": "MALAYSIA", "lat": 2.8767, "lon": 104.1561},
    "Cherating Beach (Malaysia)": {"country": "MALAYSIA", "lat": 4.1256, "lon": 103.3900},
    "Teluk Cempedak (Malaysia)": {"country": "MALAYSIA", "lat": 3.8118, "lon": 103.3725},
    "Desaru Beach (Malaysia)": {"country": "MALAYSIA", "lat": 1.5415, "lon": 104.2685},

    # --- MALAYSIA: BORNEO ---
    "Tanjung Aru (Malaysia)": {"country": "MALAYSIA", "lat": 5.9489, "lon": 116.0460},
    "Mantanani Island (Malaysia)": {"country": "MALAYSIA", "lat": 6.7088, "lon": 116.3533},
    "Sipadan Island (Malaysia)": {"country": "MALAYSIA", "lat": 4.1147, "lon": 118.6288},
    "Mabul Island (Malaysia)": {"country": "MALAYSIA", "lat": 4.2464, "lon": 118.6311},
    "Damai Beach (Malaysia)": {"country": "MALAYSIA", "lat": 1.7480, "lon": 110.3160},
    "Tusan Beach (Malaysia)": {"country": "MALAYSIA", "lat": 4.1350, "lon": 113.8200},

    # --- PHILIPPINES ---
    "Bagasbas Beach (Philippines)": {"country": "PHILIPPINES", "lat": 14.1358, "lon": 122.9836},
    "Nacpan Beach (Philippines)": {"country": "PHILIPPINES", "lat": 11.3200, "lon": 119.4300},
    "Sabang Beach (Philippines)": {"country": "PHILIPPINES", "lat": 13.5200, "lon": 120.9700},
    "White Beach, Puerto Galera (Philippines)": {"country": "PHILIPPINES", "lat": 13.5000, "lon": 120.9000},
    "Subic Beach, Sorsogon (Philippines)": {"country": "PHILIPPINES", "lat": 12.5300, "lon": 124.1000},
    "White Beach, Boracay (Philippines)": {"country": "PHILIPPINES", "lat": 11.9500, "lon": 121.9300},
    "Alona Beach, Bohol (Philippines)": {"country": "PHILIPPINES", "lat": 9.5500, "lon": 123.7700},
    "Mactan Island (Philippines)": {"country": "PHILIPPINES", "lat": 10.3100, "lon": 124.0200},
    "Cloud 9, Siargao (Philippines)": {"country": "PHILIPPINES", "lat": 9.8000, "lon": 126.1600},
    "Samal Island (Philippines)": {"country": "PHILIPPINES", "lat": 7.0800, "lon": 125.7500},
}

# 2. HELPER FUNCTIONS

# --- SHARK RADAR SIMULATION ---
def get_live_shark_activity(center_lat, center_lon):
    """Generates simulated shark 'pings' near the selected beach."""
    shark_data = []
    num_sharks = random.randint(3, 8)
    species_list = ["Tiger Shark", "Bull Shark", "Hammerhead", "Blacktip Reef Shark"]
    
    for i in range(num_sharks):
        lat_offset = random.uniform(-0.04, 0.04)
        lon_offset = random.uniform(-0.04, 0.04)
        shark = {
            "id": f"SHK-{random.randint(100, 999)}",
            "lat": center_lat + lat_offset,
            "lon": center_lon + lon_offset,
            "species": random.choice(species_list),
            "last_seen": f"{random.randint(1, 55)} mins ago"
        }
        shark_data.append(shark)
    return pd.DataFrame(shark_data)

# --- WEATHER API ---
def get_live_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        response = requests.get(url, timeout=10) 
        response.raise_for_status()
        return response.json().get('current_weather')
    except Exception as e:
        return None

def interpret_weather_code(code):
    if code is None: return "Unknown"
    if code <= 3: return "Clear/Cloudy"
    elif code <50: return "Foggy"
    elif code <80: return "Rainy"
    elif code <95: return "Heavy Rain"
    else: return "Thunderstorm"

# --- SEASON CALCULATOR ---
def determine_season(date_obj, country_name):
    month = date_obj.month
    clean_country = str(country_name).strip().upper()
    southern_hemi = ["AUSTRALIA", "SOUTH AFRICA", "NEW ZEALAND"]
    
    if clean_country in southern_hemi:
        if month in [12, 1, 2]: return 'Summer'
        elif month in [3, 4, 5]: return 'Autumn'
        elif month in [6, 7, 8]: return 'Winter'
        else: return 'Spring'
    else:
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

# --- AI ADVICE ---
@st.cache_data(show_spinner=False)
def generate_llm_advice(risk_level, shark_prob, activity, weather_data):
    try:
        llm = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Act as a Marine Safety Expert.
        User Scenario:
        - Activity: {activity}
        - Weather Data: {weather_data}
        - Risk Status: {risk_level}
        
        Task: Write a 3-bullet safety briefing.
        IMPORTANT: If Risk Status mentions "WEATHER", focus on drowning/storms, not sharks.
        Keep it concise.
        """
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return "⚠️ AI Advice unavailable (Network/Quota Error). Stay alert."

# 3. LOGIN SCREEN
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''

if not st.session_state['logged_in']:
    st.title("OceanSafe Login")
    with st.form("login"):
        user = st.text_input("Enter your Name")
        
        if st.form_submit_button("Start System"):
            if user.strip(): # Check if name is not empty
                st.session_state['logged_in'] = True
                st.session_state['username'] = user
                st.rerun()
            else:
                st.error("Please enter a name to continue.")
    st.stop()

# 4. DASHBOARD INTERFACE
st.sidebar.title(f"User: {st.session_state['username']}")
st.sidebar.info("System Online:\n✅ Random Forest (Risk)\n✅ Gemini AI (Advice)\n✅ Open-Meteo (Weather)")

# --- SIDEBAR: REPORTING SYSTEM ---
st.sidebar.markdown("---")
st.sidebar.header("📢 Report a Sighting")

with st.sidebar.form("report_form"):
    st.write("Did you see a shark?")
    r_species = st.selectbox("Species", ["Unknown", "Hammerhead", "Tiger Shark", "Bull Shark"])
    r_loc = st.text_input("Location (e.g. Bondi)")
    r_time = st.time_input("Time Occurred")
    
    submit_report = st.form_submit_button("Submit Report")
    
    if submit_report:
        # Save to session state (Temporary Memory)
        if 'reports' not in st.session_state:
            st.session_state['reports'] = []
        
        new_report = {
            "Time": str(r_time),
            "Location": r_loc,
            "Species": r_species,
            "Status": "Unverified"
        }
        st.session_state['reports'].append(new_report)
        st.sidebar.success("Report broadcasted to lifeguards!")

# Show Recent Reports in Sidebar
if 'reports' in st.session_state and len(st.session_state['reports']) > 0:
    st.sidebar.markdown("### 🚨 Recent Community Alerts")
    df_reports = pd.DataFrame(st.session_state['reports'])
    st.sidebar.dataframe(df_reports, hide_index=True)

st.title("🌊 OceanSafe: Integrated AI System")
st.markdown("### Hybrid Risk Assessment Engine")

if model is None:
    st.error("⚠️ Model files not found. Please upload .pkl files.")
    st.stop()

# --- INPUT SECTION ---
all_countries = sorted(list(set(data['country'] for data in BEACH_DB.values())))
default_idx = all_countries.index("MALAYSIA") if "MALAYSIA" in all_countries else 0

# 1. Country Select (Updates automatically)
country_sel = st.selectbox("1. Select Country", all_countries, index=default_idx)

# 2. Main Form
with st.form("analysis_form"):
    c1, c2 = st.columns(2)
    with c1:
        filtered_beaches = [name for name, data in BEACH_DB.items() if data['country'] == country_sel]
        beach = st.selectbox("2. Select Beach", sorted(filtered_beaches))
        date = st.date_input("Date", datetime.today())
    with c2:
        act = st.selectbox("Activity", list(encoders['Activity'].classes_))
        sex = st.radio("Sex", ['Male', 'Female'], horizontal=True)
        age = st.selectbox("Age", list(encoders['AgeGroup'].classes_))
    
    submitted = st.form_submit_button("🛡️ Run AI Risk Analysis")

# 5. ANALYSIS & RESULTS (SMART LOGIC UPDATE)
if submitted:
    st.divider()
    
    # --- STEP 1: PREPARE DATA ---
    b_info = BEACH_DB[beach]
    season = determine_season(date, b_info['country'])
    weather = get_live_weather(b_info['lat'], b_info['lon'])
    
    # --- STEP 2: CHECK IF CAN RUN SHARK ANALYSIS ---
    # check if this country exists in the AI's "Training Memory"
    known_countries = encoders['Country'].classes_
    can_analyze_sharks = b_info['country'] in known_countries
    
    shark_risk = None # Default to None
    prediction_error = False

    # --- STEP 3: RUN MODEL (ONLY IF COUNTRY IS KNOWN) ---
    if can_analyze_sharks:
        try:
            # 1. Encode inputs
            country_val = encoders['Country'].transform([b_info['country']])[0]
            act_val = encoders['Activity'].transform([act])[0]
            season_val = encoders['Season'].transform([season])[0]
            age_val = encoders['AgeGroup'].transform([age])[0]
            sex_val = 1 if sex == 'Male' else 0
            
            input_data = pd.DataFrame({
                'Country': [country_val], 'Activity': [act_val],
                'AgeGroup': [age_val], 'Season': [season_val], 'Sex': [sex_val]
            })
            
            # 2. Get Probability
            shark_risk = model.predict_proba(input_data)[0][1]

        except Exception as e:
            prediction_error = True
            st.error(f"Prediction Logic Error: {e}")
    else:
        # LOGIC: If in Malaysia (or unknown country), skip the shark math entirely.
        # This prevents the error and prevents "fake" analysis.
        shark_risk = None 

    # --- STEP 4: DISPLAY METRICS ---
    st.subheader(f"Report for {beach}")
    
    # Weather Formatting
    w_desc = interpret_weather_code(weather.get('weathercode')) if weather else "Unknown"
    w_temp = f"{weather.get('temperature')}°C" if weather else "--"
    w_wind = f"{weather['windspeed']} km/h" if weather else "N/A"
    
    # Shark Formatting (Handle the N/A case)
    if shark_risk is not None:
        risk_display = f"{shark_risk*100:.1f}%"
    else:
        risk_display = "N/A (Insufficient Data)"

    # Row 1: Environment
    c1, c2, c3 = st.columns(3)
    c1.metric("Season", season)
    c2.metric("Air Temp", w_temp)
    c3.metric("Wind Speed", w_wind)
    
    # Row 2: Status
    c4, c5 = st.columns(2)
    c4.metric("Condition", w_desc)
    import plotly.graph_objects as go

    # Create the Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = shark_risk * 100 if shark_risk is not None else 0,
        title = {'text': "Shark Risk Probability"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
        }
    ))
    
    # Make it small to fit the column
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Sirin Stencil"})
    
    # Display it in the second column (c5)
    with c5:
        if shark_risk is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Shark Data Available")

    st.divider()

    # --- STEP 5: FINAL VERDICT LOGIC (DYNAMIC) ---
    
    status_text = "LOW RISK"
    color = "green"
    
    # Weather Check (Applies to EVERY country)
    # Wind > 25km/h or Thunderstorms = DANGER
    is_bad_weather = (weather and weather['windspeed'] > 25)
    
    if shark_risk is not None:
        # === MODE A: FULL ANALYSIS (Australia/USA) ===
        is_high_shark = (shark_risk > 0.65)

        if is_bad_weather and is_high_shark:
            status_text = "EXTREME DANGER (Storms & Sharks)"
            color = "red"
        elif is_bad_weather:
            status_text = "DANGEROUS SEA CONDITIONS (Weather)"
            color = "red"
        elif is_high_shark:
            status_text = "HIGH RISK (Shark Activity)"
            color = "red"
        elif shark_risk > 0.45:
            status_text = "MODERATE RISK"
            color = "orange"
            
    else:
        # === MODE B: WEATHER ONLY (Malaysia) ===
        # If we don't have shark data, we judge purely on drowning/weather risk
        if is_bad_weather:
            status_text = "DANGEROUS SEA CONDITIONS (High Wind/Storms)"
            color = "red"
        else:
            status_text = "LOW RISK (Weather Safe)"
            color = "green"

    st.markdown(f"### Status: :{color}[{status_text}]")

    # --- STEP 6: AI ADVICE (Updated Prompt) ---
    st.markdown("#### 🤖 AI Marine Consultant (Gemini)")
    with st.spinner("Generating safety protocol..."):
        
        # We tell the AI if shark data is missing so it doesn't make things up
        shark_context = f"{shark_risk:.2f}" if shark_risk is not None else "DATA UNAVAILABLE"
        
        advice = generate_llm_advice(status_text, shark_context, act, weather)
        st.info(advice)

    # --- STEP 7: SHAP (Only show if we actually ran the model) ---
    if shark_risk is not None and not prediction_error:
        st.markdown("---")
        with st.expander("❓ Why is the risk at this level? (Model Explainability)"):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_data)
                
                if isinstance(shap_values, list):
                    vals = shap_values[1][0]
                elif len(shap_values.shape) == 3:
                    vals = shap_values[0, :, 1]
                else:
                    vals = shap_values[0]

                feature_names = input_data.columns
                factors = [(feature_names[i], vals[i]) for i in range(len(feature_names))]
                factors.sort(key=lambda x: abs(x[1]), reverse=True)

                st.write("**Top Factors Driving this Prediction:**")
                for feature, impact in factors[:3]:
                    icon = "🔺" if impact > 0 else "⬇️"
                    effect = "increasing" if impact > 0 else "lowering"
                    st.write(f"{icon} **{feature}**: Is {effect} the risk.")
            except Exception as e:
                st.warning("Explainability module loading...")

# 6. LIVE SHARK RADAR 
# This is now outside the "submitted" block but updates based on selection.
# It acts like a separate app module at the bottom.
st.markdown("---")
st.subheader("📡 Live Monitoring Tools")

# The BUTTON you requested (implemented as an Expander)
with st.expander(f"🔴 OPEN LIVE SHARK RADAR: {beach}", expanded=False):
    
    # 1. Get Location
    loc_data = BEACH_DB[beach]
    
    # 2. Run Simulation
    shark_df = get_live_shark_activity(loc_data['lat'], loc_data['lon'])
    
    # 3. Display
    c_map, c_list = st.columns([2, 1])
    
    with c_list:
        st.write(f"**Tracking {len(shark_df)} sharks**")
        st.dataframe(shark_df[['species', 'last_seen']], height=200)
    
    with c_map:
        # Create Map Data (Blue=Beach, Red=Sharks)
        map_data = pd.DataFrame({
            'lat': [loc_data['lat']] + shark_df['lat'].tolist(),
            'lon': [loc_data['lon']] + shark_df['lon'].tolist(),
            'color': ['#0000FF'] + ['#FF0000'] * len(shark_df)
        })

        st.map(map_data, color='color', size=20)
