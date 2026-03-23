from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import math
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Care4HIV ML Backend API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
BASE_DRIVE_PATH = "models"

try:
    # 1. HIV Risk Prediction
    risk_model = joblib.load(os.path.join(BASE_DRIVE_PATH, "risk_prediction_model.pkl"))
    risk_scaler = joblib.load(os.path.join(BASE_DRIVE_PATH, "risk_scaler.pkl"))
    risk_encoders = joblib.load(os.path.join(BASE_DRIVE_PATH, "risk_label_encoders.pkl"))

    # 2. Emotion Detection
    emotion_model = joblib.load(os.path.join(BASE_DRIVE_PATH, "emotion_model.pkl"))
    tfidf_vectorizer = joblib.load(os.path.join(BASE_DRIVE_PATH, "tfidf_vectorizer.pkl"))

    # 3. Patient Treatment Risk
    patient_model = joblib.load(os.path.join(BASE_DRIVE_PATH, "india_treatment_model.pkl"))
    patient_scaler = joblib.load(os.path.join(BASE_DRIVE_PATH, "india_scaler.pkl"))
    patient_encoders = joblib.load(os.path.join(BASE_DRIVE_PATH, "india_label_encoders.pkl"))

    # 4. Nearest Health Centers (CSV data)
    csv_path = os.path.join(BASE_DRIVE_PATH, "health_centers_clean.csv")
    if os.path.exists(csv_path):
        centers_df = pd.read_csv(csv_path)
        print("✅ Health centers data loaded successfully!")
    else:
        print(f"⚠️ Warning: {csv_path} not found. Healthcare centers feature will be disabled.")

    # 5. Chatbot Data & Vectorizer
    with open(os.path.join(BASE_DRIVE_PATH, "chatbot_data.json"), "r") as f:
        chatbot_data = json.load(f)
    
    # Flatten patterns for vectorization
    all_patterns = []
    pattern_to_intent = []
    for intent in chatbot_data['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(pattern)
            pattern_to_intent.append(intent)
            
    vectorizer = TfidfVectorizer()
    X_patterns = vectorizer.fit_transform(all_patterns)

    print("✅ All models, datasets, and chatbot logic loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load models. Error: {e}")

# --- HELPER FUNCTIONS ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * 6371 * math.asin(math.sqrt(a))

# --- REQUEST MODELS (Pydantic) ---

class HIVRiskRequest(BaseModel):
    age: int
    marital_status: str
    std: str
    education: str
    hiv_test_past_year: str
    aids_education: str
    places_seeking_partners: str
    sexual_orientation: str
    drug_taking: str

class EmotionRequest(BaseModel):
    text: str

class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    top_n: int = 5

class ChatRequest(BaseModel):
    message: str

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "Care4HIV API is running!"}

@app.post("/predict/hiv-risk")
def predict_hiv_risk(req: HIVRiskRequest):
    try:
        data = {
            'Age': [req.age],
            'Marital Staus': [req.marital_status],
            'STD': [req.std],
            'Educational Background': [req.education],
            'HIV TEST IN PAST YEAR': [req.hiv_test_past_year],
            'AIDS education': [req.aids_education],
            'Places of seeking sex partners': [req.places_seeking_partners],
            'SEXUAL ORIENTATION': [req.sexual_orientation],
            'Drug- taking': [req.drug_taking]
        }
        df = pd.DataFrame(data)

        for col in df.select_dtypes(include='object').columns:
            if col in risk_encoders:
                le = risk_encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

        df_scaled = risk_scaler.transform(df)
        prediction = risk_model.predict(df_scaled)[0]
        result = risk_encoders['Result'].inverse_transform([prediction])[0]

        return {"prediction": result, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-emotion")
def detect_emotion(req: EmotionRequest):
    try:
        text_tfidf = tfidf_vectorizer.transform([req.text])
        prediction = emotion_model.predict(text_tfidf)[0]
        return {"emotion": prediction, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nearest-centers")
def get_nearest_centers(req: LocationRequest):
    try:
        if 'centers_df' not in globals() or centers_df is None:
            raise HTTPException(
                status_code=503, 
                detail="Health centers data not loaded on server. Please ensure 'health_centers_clean.csv' is in the 'models' folder."
            )
        
        df = centers_df.copy()
        df['distance_km'] = df.apply(
            lambda row: haversine(req.latitude, req.longitude, row['Latitude'], row['Longitude']),
            axis=1
        )
        
        nearest = df.nsmallest(req.top_n, 'distance_km')
        results = nearest[['Facility Name', 'Facility Type', 'State Name', 'District Name', 'Latitude', 'Longitude', 'distance_km']].to_dict(orient='records')
        
        return {"centers": results, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chatbot")
def chatbot_response(req: ChatRequest):
    try:
        # Vectorize user message
        user_vec = vectorizer.transform([req.message.lower()])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vec, X_patterns).flatten()
        best_match_idx = np.argmax(similarities)
        max_sim = similarities[best_match_idx]
        
        # Threshold for relevant answers
        if max_sim > 0.7:
            intent = pattern_to_intent[best_match_idx]
            response = random.choice(intent['responses'])
            return {"response": response, "status": "success"}
        else:
            defaults = [
                "I can only answer questions related to HIV, testing, prevention, and treatment. Please ask me something related to HIV. 🌿",
                "I'm specialized in HIV-related topics only. Try asking about HIV symptoms, testing, prevention, or treatment.",
                "That's outside my area. I'm here to help with HIV questions only. What would you like to know about HIV? 💙"
            ]
            return {"response": random.choice(defaults), "status": "success"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
