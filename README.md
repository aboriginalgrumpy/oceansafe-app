# OceanSafe: AI-Powered Coastal Risk Assessment

**OceanSafe** is an integrated safety dashboard designed to predict shark attack risks and provide real-time lifeguard advice for beachgoers. It combines historical data analysis with generative AI and Computer Vision to offer a modern approach to marine safety.

###  Key Features

* **Hybrid AI Analysis:** Uses a Random Forest model for statistical risk prediction (with SHAP explainability) and Google Gemini (LLM) for context-aware safety advice.
* **Computer Vision (Shark Egg Scanner):** New "Mermaid's Purse" scanner allows users to upload photos of shark egg cases found on the beach. The AI identifies the species and checks if the egg is hatched or viable.
* **Smart Season & Region Logic:** Automatically detects climate context (e.g., "Wet Season/Monsoon" for tropical regions vs. "Summer" for temperate zones) and filters beaches by specific regions.
* **Recency Decay Algorithm:** The risk engine now accounts for time; beaches with no recent attacks (e.g., >5 years) receive a mathematically lower risk score.
* **Visual Risk Gauge:** Interactive dashboard featuring a speedometer-style risk probability gauge.
* **Live Shark Radar (Simulated):** A geospatial visualisation tool demonstrating how real-time tracking data appears for lifeguards.
* **Crowdsourced Sighting Reports:** Allows users to log sightings, demonstrating community-driven safety data collection.

---

### Technology Stack

* **Frontend:** Streamlit (Python)
* **AI / LLM:** Google Gemini API (Multimodal: Text & Vision)
* **Machine Learning:** Scikit-Learn (Random Forest), SHAP (Explainability)
* **Image Processing:** Pillow (PIL)
* **Data Visualisation:** Plotly, Pandas, Matplotlib
* **Deployment:** Streamlit Community Cloud

---

### ⚙️ Installation & Setup

**1. Clone this repository**
```bash
git clone [https://github.com/aboriginalgrumpy/oceansafe-app](https://github.com/aboriginalgrumpy/oceansafe-app)
cd oceansafe-ai
```
**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
**3. Run the app**
```bash
streamlit run app.py
```

### How to use
1. Login: The system uses a frictionless login. Simply enter your Name to access the dashboard
2. Select Location:
   -    Select Country and Region
   -    *note:* Selecting Malaysia will trigger the "Daata Sparsity" logic (Weather-only risk), while contries like Australia trigger the full Random Forest shark model
3. Run Analysis: Click "Run AI Risk Analysis" to see the Risk Gauge, Recency Decay factors and AI Advice
4. View Radar: Scroll down and click OPEN LIVE SHARK RADAR to see the map simulation
5. Scan an Egg Case:
   -    Scroll to the "Community Sighting Log"
   -    Click the "Scan Egg Case" tab.
   -    Upload an image of a shark egg (Mermaid's Purse).
   -    Click "Identify Species" to see the Computer Vision analysis.
  
### Academic Disclosure
- Shark Radar Simulation: The "real-time shark activit" module uses a synthetic data generator to simulate API responses from OCEARCH. This ensures the application remains functional and testable during the examination period without relying on unstable external API connections.
- Recency Logic: The application applies a mathematical decay factor ( 1 / (1 + 0.15 * Years_Gap) ) to historical risk scores to prevent outdated data from causing false alarms.
