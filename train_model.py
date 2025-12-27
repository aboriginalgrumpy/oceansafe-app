# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# 1. LOAD DATA
print("Loading dataset...")
try:
    # Use standard encoding to avoid "255 columns" error
    df = pd.read_csv("Shark_Attack2.csv", encoding="latin1")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# 2. CLEANING
print("Cleaning data...")
df = df.dropna(axis=1, how='all')

# Clean Fatal (Target)
df['Fatal'] = df['Fatal Y/N'].astype(str).str.strip().str.upper().map({'Y':1, 'N':0})
df.loc[df['Fatal'].isna() & df['Injury'].astype(str).str.contains('fatal', case=False), 'Fatal'] = 1
df = df.dropna(subset=['Fatal'])

# Clean Age
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df = df.dropna(subset=['Age'])
bins = [0, 12, 19, 35, 60, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Elderly']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Clean Sex
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df = df.dropna(subset=['Sex'])

# Clean Season
southern_hemi = ["AUSTRALIA", "SOUTH AFRICA", "NEW ZEALAND", "BRAZIL", "FIJI", "PAPUA NEW GUINEA", "SAMOA", "FRENCH POLYNESIA"]

def get_season(row):
    m = pd.to_datetime(row['Date'], errors='coerce').month
    if pd.isna(m): return 'Unknown'
    
    country = str(row['Country']).upper()
    is_southern = country in southern_hemi
    
    if is_southern:
        if m in [12, 1, 2]: return 'Summer'
        elif m in [3, 4, 5]: return 'Autumn'
        elif m in [6, 7, 8]: return 'Winter'
        else: return 'Spring'
    else:
        if m in [12, 1, 2]: return 'Winter'
        elif m in [3, 4, 5]: return 'Spring'
        elif m in [6, 7, 8]: return 'Summer'
        else: return 'Autumn'

df['Season'] = df.apply(get_season, axis=1)

# --- CRITICAL FIX: EXPANDED LISTS ---
# We increased this from 15 to 20 to ensure 'Foil Boarding' and 'Kayaking' are included
top_activities = list(df['Activity'].value_counts().nlargest(20).index)

# FORCE "Foil Boarding" to be included even if it's rare
if "Foil Boarding" in df['Activity'].unique():
    if "Foil Boarding" not in top_activities:
        top_activities.append("Foil Boarding")

df['Activity'] = df['Activity'].where(df['Activity'].isin(top_activities), 'Other')

# Clean Country (Increased to 30 to include Samoa, Columbia, etc.)
top_countries = df['Country'].value_counts().nlargest(30).index
df['Country'] = df['Country'].where(df['Country'].isin(top_countries), 'Other')

# 3. ENCODING & SAVING
print("Encoding features and saving translators...")
feature_cols = ['Country', 'Activity', 'AgeGroup', 'Season'] 
model_df = df[feature_cols + ['Sex', 'Fatal']].dropna()

encoders = {}
for col in feature_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le 

joblib.dump(encoders, 'oceansafe_encoders.pkl')

# 4. TRAINING
print("Training the AI model...")
X = model_df[feature_cols + ['Sex']]
y = model_df['Fatal']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, 'oceansafe_model.pkl')

print("✅ SUCCESS! Model updated.")
print(f"Model Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.2f}")
print("Check your App: 'Foil Boarding' should now be in the list!")