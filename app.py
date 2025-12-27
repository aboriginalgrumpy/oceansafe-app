# app.py
import streamlit as st
import pandas as pd
import joblib
import requests
import google.generativeai as genai # NEW LIBRARY
from datetime import datetime
import os

# ==========================================
# 1. SETUP & ASSETS
# ==========================================
# CONFIGURE YOUR API KEY HERE
# In a real job, use secrets, but for CAIE demo, you can paste it here:
GOOGLE_API_KEY = "AIzaSyBUxwGKxYEwTV-U3G4BzSjPz7UjsXMj17A"
genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('oceansafe_model.pkl')
        encoders = joblib.load('oceansafe_encoders.pkl')
        return model, encoders
    except FileNotFoundError:
        return None, None

model, encoders = load_assets()

BEACH_DB = {
    "Bondi Beach (Australia)": {"country": "AUSTRALIA", "lat": -33.8915, "lon": 151.2767},
    "Gold Coast (Australia)": {"country": "AUSTRALIA", "lat": -28.0167, "lon": 153.4000},
    "New Smyrna Beach (USA)": {"country": "USA", "lat": 29.0258, "lon": -80.9270},
    "Maui (USA)": {"country": "USA", "lat": 20.7984, "lon": -156.3319},
    "Jeffreys Bay (South Africa)": {"country": "SOUTH AFRICA", "lat": -34.0333, "lon": 24.9167},
    "Durban (South Africa)": {"country": "SOUTH AFRICA", "lat": -29.8587, "lon": 31.0218},
    "Piha Beach (New Zealand)": {"country": "NEW ZEALAND", "lat": -36.9536, "lon": 174.4706},
}

# ==========================================
# 2. HELPER FUNCTIONS (API + LLM)
# ==========================================
def get_live_weather(lat, lon):
    # API call to Open-Meteo
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    try:
        # INCREASED TIMEOUT to 10 seconds (Fixes N/A on slow wifi)
        response = requests.get(url, timeout=10) 
        response.raise_for_status() # Check for 404/500 errors
        return response.json().get('current_weather')
    except requests.exceptions.Timeout:
        print("⚠️ Weather API Timed Out (Your internet might be slow)")
        return None
    except Exception as e:
        print(f"⚠️ Weather API Error: {e}")
        return None

def determine_season(date_obj, country_name):
    month = date_obj.month
    clean_country = str(country_name).strip().upper()
    southern_hemi = ["AUSTRALIA", "SOUTH AFRICA", "NEW ZEALAND", "BRAZIL", "SAMOA", "FIJI", "FRENCH POLYNESIA"]
    
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

def generate_llm_advice(risk_level, shark_prob, activity, weather_data):
    # This function satisfies the "LLM Functionality" requirement
    try:
        llm = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Act as a Marine Safety Expert. 
        User Scenario:
        - Activity: {activity}
        - Calculated Shark Risk: {shark_prob*100:.1f}% ({risk_level})
        - Weather: {weather_data}
        
        Task: Write a concise 3-bullet point safety briefing. 
        If risk is high, explain why and give a survival tip.
        If weather is bad, warn about drowning.
        Keep it professional and urgent.
        """
        
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f" Debug Error: {e}"
def interpret_weather_code(code):
    if code is None: return "Unknown"
    if code <= 3: return "Clear/Cloudy"
    elif code <50: return "Foggy"
    elif code <80: return "Rainy"
    elif code <95: return "Heavy Rain"
    else: return "Thunderstorm"

# ==========================================
# 3. PAGE CONFIG & LOGIN
# ==========================================
st.set_page_config(page_title="OceanSafe AI", page_icon="🦈")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = ''

if not st.session_state['logged_in']:
    st.title("🔒 OceanSafe Login")
    with st.form("login"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.form_submit_button("Login") and user:
            st.session_state['logged_in'] = True
            st.session_state['username'] = user
            st.rerun()
    st.stop()

# ==========================================
# 4. MAIN APP
# ==========================================
st.sidebar.title(f"User: {st.session_state['username']}")
st.sidebar.info("System Components:\n1. Random Forest (Risk Model)\n2. Google Gemini (Advice Bot)\n3. Open-Meteo (Live Data)")

st.title("🌊 OceanSafe: Integrated AI System")
st.markdown("### Hybrid Risk Assessment Engine")

if model is None:
    st.error("Missing Model Files. Run train_model.py!")
    st.stop()

with st.form("analysis_form"):
    c1, c2 = st.columns(2)
    with c1:
        beach = st.selectbox("Beach", list(BEACH_DB.keys()))
        date = st.date_input("Date", datetime.today())
    with c2:
        act = st.selectbox("Activity", list(encoders['Activity'].classes_))
        sex = st.radio("Sex", ['Male', 'Female'], horizontal=True)
        age = st.selectbox("Age", list(encoders['AgeGroup'].classes_))
    
    submitted = st.form_submit_button("Run AI Analysis")

if submitted:
    st.divider()
    
    # 1. PREPARE DATA
    b_info = BEACH_DB[beach]
    season = determine_season(date, b_info['country'])
    weather = get_live_weather(b_info['lat'], b_info['lon'])
    
    # 2. RANDOM FOREST PREDICTION (The "Other AI Component")
    try:
        country_val = encoders['Country'].transform([b_info['country']])[0]
        act_val = encoders['Activity'].transform([act])[0]
        season_val = encoders['Season'].transform([season])[0]
        age_val = encoders['AgeGroup'].transform([age])[0]
        sex_val = 1 if sex == 'Male' else 0
        
        input_data = pd.DataFrame({
            'Country': [country_val], 'Activity': [act_val],
            'AgeGroup': [age_val], 'Season': [season_val], 'Sex': [sex_val]
        })
        shark_risk = model.predict_proba(input_data)[0][1]
    except:
        st.error("Model Error")
        st.stop()

    # 3. DISPLAY METRICS
    st.subheader(f"Report for {beach}")
    
    # Get text description for weather (e.g., "Foggy")
    weather_desc = "Unknown"
    temp = "--"
    if weather:
        weather_desc = interpret_weather_code(weather.get('weathercode'))
        temp = weather.get('temperature')

    # Create 4 columns for Weather Data
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season", season)
    m2.metric("Air Temp", f"{temp}°C")
    m3.metric("Wind Speed", f"{weather['windspeed']} km/h" if weather else "N/A")
    m4.metric("Condition", weather_desc)

    st.divider() # Visual separator

    # Show Shark Risk big and bold
    # We use st.metric directly so it takes up the full width
    st.metric("🦈 Shark Risk Probability", f"{shark_risk*100:.1f}%")
    
    # --- DELETE THE LINES BELOW THAT CAUSED THE ERROR ---
    # c1 = st.columns(3)  <-- Delete this
    # c1.metric(...)      <-- Delete this

    # 4. DETERMINE VERDICT
    risk_status = "LOW"
    color = "green"
    if shark_risk > 0.65 or (weather and weather['windspeed'] > 25):
        risk_status = "HIGH"
        color = "red"
    elif shark_risk > 0.45:
        risk_status = "MODERATE"
        color = "orange"

    st.markdown(f"### Status: :{color}[{risk_status} RISK]")

    # 5. GENERATE LLM ADVICE (Crucial for CAIE Rubric)
    st.markdown("#### 🤖 AI Marine Consultant (Gemini)")
    with st.spinner("Generating survival protocol..."):
        # We pass the data to the LLM
        advice = generate_llm_advice(risk_status, shark_risk, act, weather)
        st.info(advice)