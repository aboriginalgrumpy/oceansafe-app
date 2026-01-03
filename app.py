import streamlit as st
import pandas as pd
import joblib
import requests
import google.generativeai as genai
from datetime import datetime, date
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import random

# --- RECENCY LOGIC FUNCTION ---
def calculate_recency_decay(last_incident_year):
    """
    Reduces risk score if the last attack was a long time ago.
    Formula: Decay = 1 / (1 + 0.15 * Years_Gap)
    """
    if last_incident_year is None:
        return 1.0 # If unknown, assume high caution (no decay)
        
    current_year = 2025 # Or use datetime.now().year
    years_gap = current_year - last_incident_year
    
    # If incident was very recent (0-2 years ago), Keep Risk High (100%)
    if years_gap <= 2:
        return 1.0
        
    # Example: 10 year gap -> Risk reduced by ~50%
    decay_factor = 1 / (1 + (0.15 * years_gap)) 
    return decay_factor

# 1. SETUP & CONFIGURATION

st.set_page_config(
    page_title="OceanSafe AI",
    page_icon="loglow.png",
    layout="centered")
st.markdown("""
    <style>
    /* 1. Import Sirin Stencil from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Sirin+Stencil&display=swap');

    /* 2. (Titles, Headers, and Text) */
    html, body, [class*="css"] {
        font-family: 'Sirin Stencil', sans-serif;
    }

    /* 3. Specific Styling for the Big Title */
    h1 {
        font-family: 'Sirin Stencil', sans-serif;
        color: #00B4D8; /* Cyan Color */
        text-shadow: 0 0 5px #00B4D8; /* Subtle Glow */
        text-align: center;
        font-size: 4rem; /* Make it HUGE */
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
    # --- AUSTRALIA ---
    "Bondi Beach (Australia)": {
        "country": "AUSTRALIA", "region": "New South Wales", 
        "lat": -33.8915, "lon": 151.2767, "last_year": 2022},
    "Gold Coast (Australia)": {
        "country": "AUSTRALIA", "region": "Queensland", 
        "lat": -28.0167, "lon": 153.4000, "last_year": 2024},

    # --- USA ---
    "New Smyrna Beach (USA)": {
        "country": "USA", "region": "Florida", 
        "lat": 29.0258, "lon": -80.9270, "last_year": 2023},
    "Maui (USA)": {
        "country": "USA", "region": "Hawaii", 
        "lat": 20.7984, "lon": -156.3319, "last_year": 2023},

    # --- SOUTH AFRICA ---
    "Jeffreys Bay (South Africa)": {
        "country": "SOUTH AFRICA", "region": "Eastern Cape", 
        "lat": -34.0333, "lon": 24.9167, "last_year": 2023},
    "Durban (South Africa)": {
        "country": "SOUTH AFRICA", "region": "KwaZulu-Natal", 
        "lat": -29.8587, "lon": 31.0218, "last_year": 2015},

    # --- NEW ZEALAND ---
    "Piha Beach (New Zealand)": {
        "country": "NEW ZEALAND", "region": "Auckland", 
        "lat": -36.9536, "lon": 174.4706, "last_year": 2020},

    # --- MALAYSIA: WEST COAST ---
    "Cenang Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Kedah (Langkawi)", "lat": 6.2913, "lon": 99.7278},
    "Tanjung Rhu (Malaysia)": {
        "country": "MALAYSIA", "region": "Kedah (Langkawi)", "lat": 6.4552, "lon": 99.8228},
    "Batu Ferringhi (Malaysia)": {
        "country": "MALAYSIA", "region": "Penang", "lat": 5.4735, "lon": 100.2452},
    "Monkey Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Penang", "lat": 5.4770, "lon": 100.1837},
    "Pantai Kerachut (Malaysia)": {
        "country": "MALAYSIA", "region": "Penang", "lat": 5.4542, "lon": 100.1770},
    "Teluk Nipah (Malaysia)": {
        "country": "MALAYSIA", "region": "Perak", "lat": 4.2307, "lon": 100.5447},
    "Coral Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Perak", "lat": 4.2374, "lon": 100.5434},
    "Port Dickson (Malaysia)": {
        "country": "MALAYSIA", "region": "Negeri Sembilan", "lat": 2.4344, "lon": 101.8546},
    "Blue Lagoon (Malaysia)": {
        "country": "MALAYSIA", "region": "Negeri Sembilan", "lat": 2.4139, "lon": 101.8550},

    # --- MALAYSIA: EAST COAST ---
    "Long Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Terengganu", "lat": 5.9224, "lon": 102.7214},
    "Coral Bay (Malaysia)": {
        "country": "MALAYSIA", "region": "Terengganu", "lat": 5.9173, "lon": 102.7153},
    "Pasir Panjang (Malaysia), Redang": {
        "country": "MALAYSIA", "region": "Terengganu", "lat": 5.7728, "lon": 103.0336},
    "Juara Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Pahang", "lat": 2.7933, "lon": 104.2045},
    "Cherating Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Pahang", "lat": 4.1256, "lon": 103.3900},
    "Desaru Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Johor", "lat": 1.5415, "lon": 104.2685},

    # --- MALAYSIA: BORNEO ---
    "Tanjung Aru (Malaysia)": {
        "country": "MALAYSIA", "region": "Sabah", "lat": 5.9489, "lon": 116.0460},
    "Sipadan Island (Malaysia)": {
        "country": "MALAYSIA", "region": "Sabah", "lat": 4.1147, "lon": 118.6288},
    "Damai Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Sarawak", "lat": 1.7480, "lon": 110.3160},
    "Tusan Beach (Malaysia)": {
        "country": "MALAYSIA", "region": "Sarawak", "lat": 4.1350, "lon": 113.8200},

    # --- PHILIPPINES ---
    "Bagasbas Beach (Philippines)": {
        "country": "PHILIPPINES", "region": "Luzon", "lat": 14.1358, "lon": 122.9836},
    "Nacpan Beach (Philippines)": {
        "country": "PHILIPPINES", "region": "Luzon (Palawan)", "lat": 11.3200, "lon": 119.4300},
    "Sabang Beach (Philippines)": {
        "country": "PHILIPPINES", "region": "Luzon", "lat": 13.5200, "lon": 120.9700},
    "White Beach, Boracay (Philippines)": {
        "country": "PHILIPPINES", "region": "Visayas", "lat": 11.9500, "lon": 121.9300},
    "Alona Beach, Bohol (Philippines)": {
        "country": "PHILIPPINES", "region": "Visayas", "lat": 9.5500, "lon": 123.7700},
    "Cloud 9, Siargao (Philippines)": {
        "country": "PHILIPPINES", "region": "Mindanao", "lat": 9.8000, "lon": 126.1600},
    "Samal Island (Philippines)": {
        "country": "PHILIPPINES", "region": "Mindanao", "lat": 7.0800, "lon": 125.7500},
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
def get_climate_season(country, month):
    """
    Returns 'Summer/Winter' for temperate zones 
    and 'Wet/Dry Season' for tropical zones.
    """
    # 1. TROPICAL COUNTRIES (Malaysia, Philippines, Thailand, etc.)
    # Logic: Northeast Monsoon (Nov-March) is usually "Wet"
    if country in ["MALAYSIA", "PHILIPPINES", "THAILAND", "INDONESIA", "VIETNAM"]:
        if month in [11, 12, 1, 2, 3]:
            return "Wet Season (Monsoon)"
        else:
            return "Dry Season"

    # 2. SOUTHERN HEMISPHERE (Australia, South Africa, NZ, Brazil)
    # Logic: Dec/Jan is Summer
    elif country in ["AUSTRALIA", "SOUTH AFRICA", "NEW ZEALAND", "BRAZIL"]:
        if month in [12, 1, 2]: return "Summer"
        elif month in [3, 4, 5]: return "Autumn"
        elif month in [6, 7, 8]: return "Winter"
        else: return "Spring"

    # 3. NORTHERN HEMISPHERE (USA, Europe, Japan)
    # Logic: Dec/Jan is Winter
    else:
        if month in [12, 1, 2]: return "Winter"
        elif month in [3, 4, 5]: return "Spring"
        elif month in [6, 7, 8]: return "Summer"
        else: return "Autumn"

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

# --- 4. DASHBOARD INTERFACE (SIMPLIFIED) ---
st.sidebar.title(f"User: {st.session_state['username']}")
st.sidebar.info("System Online:\n✅ Random Forest (Risk)\n✅ Gemini AI (Advice)\n✅ Open-Meteo (Weather)")

if 'reports' in st.session_state and len(st.session_state['reports']) > 0:
    st.sidebar.markdown("### 🚨 Recent Community Alerts")
    df_reports = pd.DataFrame(st.session_state['reports'])
    st.sidebar.dataframe(df_reports, hide_index=True)

# --- HEADER SECTION ---
import base64
from pathlib import Path

def get_base64_image(image_path):
    try:
        img_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""
icon_html = get_base64_image("loglow.png")

st.markdown(
    f"""
    <h2 style='display: flex; align-items: center;'>
        <img src='{icon_html}' style='height: 40px; margin-right: 10px; margin-bottom: 5px;'>
        Hybrid Risk Assessment Engine
    </h2>
    """,
    unsafe_allow_html=True
)

if model is None:
    st.error("⚠️ Model files not found. Please upload .pkl files.")
    st.stop()

# 1. Select Country (Type to Search works automatically!)
all_countries = sorted(list(set(d['country'] for d in BEACH_DB.values())))
country = st.selectbox("Select Country", all_countries)

# 2. Select Region (Filters based on Country)
# We find all regions that belong to the selected country
regions_in_country = sorted(list(set(
    d.get('region', 'Unknown') for name, d in BEACH_DB.items() if d['country'] == country
)))
region = st.selectbox("Select Region", regions_in_country)

# 3. Select Beach (Filters based on Region)
beaches_in_region = sorted([
    name for name, d in BEACH_DB.items() 
    if d.get('region', 'Unknown') == region and d['country'] == country
])
beach = st.selectbox("Select Beach", beaches_in_region)

# 4. Clean Activity List (Top 10 Only)
clean_activities = [
    "Swimming", "Surfing", "Fishing", "Snorkeling", 
    "Scuba Diving", "Wading", "Body Boarding", 
    "Standing", "Kayaking", "Boogie Boarding"
]
act = st.selectbox("Activity", clean_activities)

# 5. Date & Season Logic
date_input = st.date_input("Date", value=date(2025, 12, 27))
month = date_input.month

# Call the NEW Season Function
season_str = get_climate_season(country, month)

# Display the detected season nicely
st.info(f"📍 Context: **{season_str}** detected for {country} in Month {month}.")
submitted = st.button("Run AI Risk Analysis", type="primary", use_container_width=True)

# 5. ANALYSIS & RESULTS (SMART LOGIC UPDATE)
if submitted:
    st.divider()
    
    # --- STEP 1: PREPARE DATA ---
    b_info = BEACH_DB[beach]
    model_season = determine_season(date_input, b_info['country'])
    display_season = get_climate_season(b_info['country'], date_input.month)
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
            season_val = encoders['Season'].transform([model.season])[0]
            age_val = encoders['AgeGroup'].transform([age])[0]
            sex_val = 1 if sex == 'Male' else 0
            
            input_data = pd.DataFrame({
                'Country': [country_val], 'Activity': [act_val],
                'AgeGroup': [age_val], 'Season': [season_val], 'Sex': [sex_val]
            })
            
            # 2. Get Raw Probability (Frequency Based)
            raw_risk = model.predict_proba(input_data)[0][1]

            # --- 3. APPLY RECENCY LOGIC (Time Based) ---
            # A. Check for specific Beach Data first
            beach_specific_year = b_info.get('last_year')
            
            # B. If Beach has no data, check Country Default
            COUNTRY_DEFAULTS = {
                "AUSTRALIA": 2024, "USA": 2024, "SOUTH AFRICA": 2023, 
                "NEW ZEALAND": 2022, "MALAYSIA": 1950, "PHILIPPINES": 1960,
                "THAILAND": 2000
            }
            final_last_year = beach_specific_year if beach_specific_year else COUNTRY_DEFAULTS.get(b_info['country'], 2025)
            
            # C. Calculate Decay & Final Risk
            decay = calculate_recency_decay(final_last_year)
            shark_risk = raw_risk * decay
            

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
    c1.metric("Season", display_season)
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
def get_base64_image(image_path):
    try:
        img_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""
icon_html = get_base64_image("icon_sat.png")

st.markdown(
    f"""
    <h2 style='display: flex; align-items: center;'>
        <img src='{icon_html}' style='height: 40px; margin-right: 10px; margin-bottom: 5px;'>
        Live Radar
    </h2>
    """,
    unsafe_allow_html=True
)

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

# --- 6. COMMUNITY REPORTING & EGG SCANNER ---
def get_base64_image(image_path):
    try:
        img_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    except FileNotFoundError:
        return ""
icon_html = get_base64_image("icon_report.png")

st.markdown(
    f"""
    <h2 style='display: flex; align-items: center;'>
        <img src='{icon_html}' style='height: 40px; margin-right: 10px; margin-bottom: 5px;'>
        Sighting Log
    </h2>
    """,
    unsafe_allow_html=True
)

with st.expander("📝 Report a Sighting / Scan Egg Case"):

    # Create Two Tabs: One for Sharks, One for the New Egg Feature
    tab1, tab2 = st.tabs(["🦈 Report Shark", "🥚 Scan Egg Case"])
    
    # --- TAB 1: SHARK REPORT ---
    with tab1:
        with st.form("shark_form"):
            st.write("Did you see a shark? Help keep others safe.")
            r_species = st.selectbox("Species", ["Unknown", "Great White", "Tiger Shark", "Bull Shark"])
            r_loc = st.text_input("Location (e.g. Bondi)")
            r_time = st.time_input("Time Occurred")
            
            # Full width button for mobile
            submit_shark = st.form_submit_button("Broadcast Shark Alert", use_container_width=True)
            
            if submit_shark:
                if 'reports' not in st.session_state:
                    st.session_state['reports'] = []
                
                new_report = {
                    "Time": str(r_time),
                    "Location": r_loc,
                    "Species": r_species,
                    "Status": "Unverified"
                }
                st.session_state['reports'].append(new_report)
                st.success(f"⚠️ Alert Sent: {r_species} sighted at {r_loc}!")

    # --- TAB 2: EGG SCANNER ---
    with tab2:
        st.write("Found a 'Mermaid's Purse'? Upload a photo and AI will identify it.")
        uploaded_egg = st.file_uploader("Upload Egg Photo", type=["jpg", "png", "jpeg"])
        
        if uploaded_egg is not None:
            st.image(uploaded_egg, caption="Uploaded Specimen", width=200)
            
            if st.button("🔍 Identify Species"):
                with st.spinner("Analyzing biological structures..."):
                    try:
                        # 1. Prepare Image
                        from PIL import Image
                        image = Image.open(uploaded_egg)
                        
                        # 2. Configure Model (Use the standard Flash model)
                        # The update you just did makes this work!
                        vision_model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        # 3. Ask Question
                        response = vision_model.generate_content([
                            "Identify this shark/ray egg case. Give species and status (hatched/viable).", 
                            image
                        ])
                        
                        # 4. Show Result
                        st.info("🧬 **AI Identification Result:**")
                        st.write(response.text)
                        
                    except Exception as e:
                        st.error("❌ Analysis Failed")
                        st.write(f"Technical Error: {e}")
                        st.caption("Please ensure you ran: pip install --upgrade google-generativeai")

# --- DISPLAY RECENT ALERTS (Optional: Shows list below the form) ---
if 'reports' in st.session_state and len(st.session_state['reports']) > 0:
    st.markdown("##### Recent Alerts:")
    st.dataframe(st.session_state['reports'], use_container_width=True)