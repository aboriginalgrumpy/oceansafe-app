# OceanSafe: AI-Powered Coastal Risk Assessment
**OceanSafe** is an integrated safety dashboard designed to predict shark attack risks and provide real-time lifeguard advice for beachgoers. It combines historical data analysis with generative AI to offer a modern approach to marine safety.

## Key Features:
* ** Hybrid AI Analysis:** Uses a Random Forest model for statistical risk prediction with SHAP confidentiality and Google Gemini (LLM) for context-aware safety advice
* ** Visual Risk Gauge:** Interactive dashboard featuring a speedometer-style risk probability  gauge.
* ** Live Shark Radar (Simulated):** A geospatial visualisation tool demonstrating how real-time tracking data appears for lifeguards.
* ** Crowdsourced Sighting Reports:** Allows users to log sightings, demonstrating community-driven safety data collection.
* ** Environmental Monitoring:** Integrates live weather data (Wind Speed, Temperature) via the Open-Meteo API

## Technology Stack
* **Frontend:** Streamlit (Python)
* **AI/LLM:** Google Gemini API (1.5 flash)
* **Machine Learning:** Scikit-Learn (Random Forest), SHAP (explainability)
* **Data Visualisation:** Plotly, Pandas, Matplotlb
* **Deployment:** Streamlit Community Cloud

## Installation & Setup

1. **Clone this repo**
    ```bash
    git clone [https://github.com/aboriginalgrumpy/oceansafe-app](https://github.com/aboriginalgrumpy/oceansafe-app)
    cd oceansafe-ai
    ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. *Run the app**
   ```bash
   streamlit run app.py
   ```

## How to use (for examiner)
1. **Login:** The system uses a frictionless login. Simply enter your **Name** to access the dashboard
2. **Select Location:** Choose a country from the dropdown.
   * *Note:* selecting **Malaysia** will trigger the "Data Sparsity" logic (Weather-only risk), while other countries trigger the full shark AI model.
3. **Run Analysis:** Click "Run AI Risk Analysis" to see the Risk Gauge and AI Advice.
4. **View Radar:** Scroll to the bottom and click ** OPEN LIVE SHARK RADAR ** to see the map simulation

## Academic Disclosure
**Shark Radar Simulation:** The "Real-Time Shark Activity" module uses a synthetic data generator to simulate API responses from OCEARCH. This ensures the application remains functional and testable during the examination period without relying on unstable external API connections.
