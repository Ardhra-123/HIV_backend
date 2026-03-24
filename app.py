from fastapi import FastAPI, HTTPException, UploadFile, File
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
from PIL import Image
import io
import re

try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("⚠️ pytesseract not installed. OCR features will be disabled.")

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
        text = req.text.lower()

        # ── Keyword-based emotion scoring ─────────────────────────────────
        emotion_keywords = {
            "happy": ["happy", "joy", "wonderful", "great", "amazing", "good", "fantastic",
                      "smile", "laugh", "love", "excited", "blessed", "cheerful", "proud",
                      "positive", "better", "well", "glad", "delighted", "pleasure",
                      "content", "thrilled", "elated", "celebrate", "enjoy"],
            "grateful": ["grateful", "thankful", "appreciate", "thanks", "blessed",
                         "gratitude", "fortunate", "lucky", "privilege"],
            "stressed": ["stressed", "overwhelmed", "pressure", "overworked", "exhausted",
                         "burnout", "burnt out", "too much", "can't handle", "breaking point",
                         "tension", "deadlines", "frustrated", "swamped", "hectic",
                         "overloaded", "tired", "drained"],
            "anxious": ["anxious", "worried", "nervous", "panic", "uneasy", "restless",
                        "overthinking", "can't stop thinking", "what if", "tense",
                        "on edge", "butterflies", "dread", "apprehensive", "jittery",
                        "racing thoughts", "can't sleep", "insomnia", "obsessing"],
            "sad": ["sad", "depressed", "unhappy", "miserable", "hopeless", "empty",
                    "crying", "tears", "heartbroken", "grief", "mourn", "lost",
                    "down", "blue", "low", "gloomy", "devastated", "despair",
                    "painful", "suffering", "hurting", "sorrow", "dejected"],
            "angry": ["angry", "furious", "mad", "rage", "irritated", "annoyed",
                      "frustrated", "pissed", "hate", "resentment", "bitter",
                      "hostile", "outraged", "infuriated", "livid", "agitated",
                      "fed up", "disgusted", "sick of"],
            "fearful": ["scared", "afraid", "fear", "terrified", "frightened", "phobia",
                        "horror", "nightmare", "threat", "danger", "unsafe",
                        "helpless", "vulnerable", "intimidated", "petrified"],
            "lonely": ["lonely", "alone", "isolated", "nobody", "no one", "abandoned",
                       "disconnected", "unwanted", "rejected", "left out",
                       "by myself", "friendless", "invisible", "forgotten",
                       "miss someone", "miss my", "no friends"]
        }

        # Score each emotion
        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
                    # Boost multi-word matches
                    if " " in keyword:
                        score += 1
            scores[emotion] = score

        # Find the emotion with the highest score
        best_emotion = max(scores, key=scores.get)
        best_score = scores[best_emotion]

        if best_score > 0:
            return {"emotion": best_emotion, "status": "success"}

        # Fallback to ML model if no keywords matched
        try:
            text_tfidf = tfidf_vectorizer.transform([req.text])
            prediction = emotion_model.predict(text_tfidf)[0]
            return {"emotion": prediction, "status": "success"}
        except Exception:
            return {"emotion": "neutral", "status": "success"}

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
        # Detect emotion from the message
        text_tfidf = tfidf_vectorizer.transform([req.message])
        emotion = emotion_model.predict(text_tfidf)[0]
        
        # Vectorize user message for Q&A
        user_msg = req.message.lower()
        user_vec = vectorizer.transform([user_msg])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vec, X_patterns).flatten()
        best_match_idx = np.argmax(similarities)
        max_sim = similarities[best_match_idx]
        
        print(f"💬 Chatbot: User='{user_msg}' | Emotion={emotion} | Max Similarity={max_sim:.4f}")

        if max_sim > 0.3:
            intent = pattern_to_intent[best_match_idx]
            response = random.choice(intent['responses'])
            return {
                "response": response, 
                "emotion": emotion,
                "status": "success"
            }
        else:
            defaults = [
                "I am an **HIV-specialized assistant** and can only answer questions about HIV, AIDS, testing, prevention, treatment (ART), and related health topics.\n\nPlease ask me something related to HIV. For example:\n• What is HIV?\n• How is HIV prevented?\n• What does my CD4 count mean?\n• Tell me about PrEP",
                "That question is outside my area of expertise. I'm trained **exclusively on HIV-related topics** including symptoms, testing, prevention, treatment, lab reports, and emotional support.\n\nTry asking about HIV! 🌿",
                "I can only help with HIV, AIDS, ART, testing, and related health queries. I cannot answer general or non-medical questions.\n\n💡 Try: 'What are the symptoms of HIV?' or 'Explain my viral load' 💙"
            ]
            return {
                "response": random.choice(defaults), 
                "emotion": emotion,
                "status": "success"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Lab Report Value Definitions ─────────────────────────────────────────────
LAB_MARKERS = [
    {
        "name": "CD4 Count",
        "patterns": [
            r"cd4[\s\+\-]*(?:count|cells|absolute)?[\s:=]*([\d,]+)(?:\s|$)",
            r"cd[\s]*4[\s:=]*([\d,]+)(?:\s|$)",
            r"t[\s\-]*helper[\s]*cells[\s:=]*([\d,]+)(?:\s|$)",
            r"absolute[\s]*cd4[\s:=]*([\d,]+)(?:\s|$)"
        ],
        "unit": "cells/mm³",
        "normal": "500 – 1,500",
        "interpret": lambda v: "✅ Normal immune function" if v >= 500 else ("⚠️ Moderate immune suppression — ART recommended" if v >= 200 else "🚨 Severe immune suppression (AIDS-defining) — urgent ART needed")
    },
    {
        "name": "Viral Load",
        "patterns": [
            r"viral[\s]*load[\s:=]*([\d,]+)",
            r"hiv[\s\-]*(?:1)?[\s]*rna[\s:=]*([\d,]+)",
            r"copies[/\s]*ml[\s:=]*([\d,]+)",
            r"hiv[\s]*quantitative[\s:=]*([\d,]+)",
            r"(?:result|value)[\s:=]*([\d,]+)[\s]*copies",
            r"(<[\s]*\d+)" # Target < 50 or < 20 directly if it says "Target Not Detected < 20 copies"
        ],
        "unit": "copies/mL",
        "normal": "< 50 (Undetectable)",
        "interpret": lambda v: "✅ Undetectable — U=U applies! 🎉" if float(v) < 50 else ("⚠️ Low but detectable — continue ART" if float(v) < 200 else ("⚠️ Moderate — discuss with doctor" if float(v) < 10000 else "🚨 High viral load — ART adjustment may be needed"))
    },
    {
        "name": "Hemoglobin (Hb)",
        "patterns": [
            r"h(?:ae)?moglobin[\s:=]*([\d\.]+)",
            r"\bhb\b[\s:=]*([\d\.]+)",
            r"\bhgb\b[\s:=]*([\d\.]+)"
        ],
        "unit": "g/dL",
        "normal": "12.0 – 17.5",
        "interpret": lambda v: "✅ Normal" if 12.0 <= float(v) <= 17.5 else ("⚠️ Low (Anemia) — may cause fatigue; consult doctor" if float(v) < 12.0 else "⚠️ Elevated — consult doctor")
    },
    {
        "name": "WBC (White Blood Cells)",
        "patterns": [
            r"wbc[\s:=]*([\d\.]+)",
            r"white[\s]*blood[\s]*cell[s]?[\s:=]*([\d\.]+)",
            r"total[\s]*w(?:hite)?[\s]*b(?:lood)?[\s]*c(?:ell)?[\s:=]*([\d\.]+)",
            r"tlc[\s:=]*([\d\.]+)"
        ],
        "unit": "× 10³/µL",
        "normal": "4.0 – 11.0",
        "interpret": lambda v: "✅ Normal" if 4.0 <= float(v) <= 11.0 else ("⚠️ Low (Leukopenia) — immune system may be weakened" if float(v) < 4.0 else "⚠️ Elevated — possible infection or inflammation")
    },
    {
        "name": "Platelet Count",
        "patterns": [
            r"platelet[s]?[\s:=]*([\d\.]+)",
            r"\bplt\b[\s:=]*([\d\.]+)"
        ],
        "unit": "× 10³/µL",
        "normal": "150 – 400",
        "interpret": lambda v: "✅ Normal" if 150 <= float(v) <= 400 else ("⚠️ Low (Thrombocytopenia) — increased bleeding risk" if float(v) < 150 else "⚠️ Elevated — consult doctor")
    },
    {
        "name": "ALT (Liver Function)",
        "patterns": [
            r"\balt\b[\s:=]*([\d\.]+)",
            r"\bsgpt\b[\s:=]*([\d\.]+)",
            r"alanine[\s]*transaminase[\s:=]*([\d\.]+)"
        ],
        "unit": "U/L",
        "normal": "7 – 56",
        "interpret": lambda v: "✅ Normal" if 7 <= float(v) <= 56 else ("⚠️ Elevated — possible liver stress from ART; consult doctor" if float(v) > 56 else "ℹ️ Low — usually not clinically significant")
    },
    {
        "name": "AST (Liver Function)",
        "patterns": [
            r"\bast\b[\s:=]*([\d\.]+)",
            r"\bsgot\b[\s:=]*([\d\.]+)",
            r"aspartate[\s]*transaminase[\s:=]*([\d\.]+)"
        ],
        "unit": "U/L",
        "normal": "10 – 40",
        "interpret": lambda v: "✅ Normal" if 10 <= float(v) <= 40 else ("⚠️ Elevated — possible liver issue; monitor closely" if float(v) > 40 else "ℹ️ Low — usually not significant")
    },
    {
        "name": "Creatinine (Kidney Function)",
        "patterns": [
            r"creatinine[\s:=]*([\d\.]+)",
            r"serum[\s]*creatinine[\s:=]*([\d\.]+)"
        ],
        "unit": "mg/dL",
        "normal": "0.7 – 1.3",
        "interpret": lambda v: "✅ Normal kidney function" if 0.7 <= float(v) <= 1.3 else ("⚠️ Elevated — possible kidney issue (some ART drugs affect kidneys)" if float(v) > 1.3 else "ℹ️ Low — usually not significant")
    },
    {
        "name": "Total Cholesterol",
        "patterns": [
            r"total[\s]*cholesterol[\s:=]*([\d\.]+)",
            r"(?<!hdl )(?<!ldl )cholesterol[\s:=]*([\d\.]+)"
        ],
        "unit": "mg/dL",
        "normal": "< 200",
        "interpret": lambda v: "✅ Desirable" if float(v) < 200 else ("⚠️ Borderline high" if float(v) < 240 else "🚨 High — dietary changes and monitoring needed")
    },
    {
        "name": "Blood Sugar (Glucose)",
        "patterns": [
            r"(?:fasting)?[\s]*(?:blood)?[\s]*(?:sugar|glucose)[\s]*(?:fasting)?[\s:=]*([\d\.]+)",
            r"\bfbs\b[\s:=]*([\d\.]+)",
            r"\brbs\b[\s:=]*([\d\.]+)"
        ],
        "unit": "mg/dL",
        "normal": "70 – 100 (fasting)",
        "interpret": lambda v: "✅ Normal" if 70 <= float(v) <= 100 else ("⚠️ Pre-diabetic range (if fasting)" if float(v) <= 126 else "🚨 Diabetic range — consult doctor")
    }
]

def _parse_lab_values(text: str) -> list:
    """Extract lab values from OCR text using regex patterns."""
    results = []
    text_lower = text.lower()
    for marker in LAB_MARKERS:
        for pattern in marker["patterns"]:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    raw = match.group(1).replace(",", "")
                    value = float(raw)
                    interpretation = marker["interpret"](value)
                    results.append({
                        "name": marker["name"],
                        "value": value,
                        "unit": marker["unit"],
                        "normal_range": marker["normal"],
                        "interpretation": interpretation
                    })
                except (ValueError, IndexError):
                    pass
                break  # Stop after first match for this marker
    return results

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        width, height = image.size
        img_format = image.format or "Unknown"

        extracted_text = ""
        ocr_success = False

        # Attempt OCR if pytesseract is available
        if pytesseract is not None:
            try:
                extracted_text = pytesseract.image_to_string(image)
                ocr_success = bool(extracted_text.strip())
                print(f"📄 OCR extracted {len(extracted_text)} characters")
            except Exception as ocr_err:
                print(f"⚠️ OCR failed: {ocr_err}")

        if ocr_success and len(extracted_text.strip()) > 20:
            # ── Lab Report Analysis ──────────────────────────────────────
            lab_values = _parse_lab_values(extracted_text)

            response = "📄 **Lab Report Analysis**\n\n"

            if lab_values:
                response += "I detected the following values in your report:\n\n"
                for item in lab_values:
                    response += f"🔬 **{item['name']}:** {item['value']} {item['unit']}\n"
                    response += f"   Normal range: {item['normal_range']}\n"
                    response += f"   → {item['interpretation']}\n\n"
            else:
                response += "I could read text from your image but could not identify specific lab values.\n\n"
                response += "📝 **Extracted text preview:**\n"
                response += f"> {extracted_text[:500]}...\n\n" if len(extracted_text) > 500 else f"> {extracted_text}\n\n"

            response += "⚠️ **Important Disclaimer:**\n"
            response += "This is an AI-assisted reading and NOT a medical diagnosis. "
            response += "Always consult your doctor for interpretation of your lab results.\n\n"
            response += "💡 **Tip:** You can also ask me about specific values — e.g., 'What does a CD4 count of 350 mean?'"

            return {
                "response": response,
                "details": {
                    "dimensions": f"{width}x{height}",
                    "format": img_format,
                    "type": "lab_report_ocr",
                    "values_found": len(lab_values),
                    "extracted_text_length": len(extracted_text)
                },
                "status": "success"
            }
        else:
            # ── Symptom/Skin Photo (Fallback) ────────────────────────────
            response = f"I've received your image ({width}x{height}, {img_format}).\n\n"
            response += "🔍 **Analysis:**\n"
            response += "I could not detect text in this image, so it may be a photo rather than a lab report.\n\n"
            response += "🌿 **If this is a photo of a skin symptom:**\n"
            response += "Common HIV-related skin conditions include:\n"
            response += "• **Maculopapular rash** — flat, red rash with small bumps (acute HIV)\n"
            response += "• **Seborrheic Dermatitis** — scaly patches on face or scalp\n"
            response += "• **Herpes Zoster (Shingles)** — painful blistering rash\n"
            response += "• **Kaposi's Sarcoma** — dark purple/brown lesions\n"
            response += "• **Oral Thrush** — white patches in mouth\n\n"
            response += "⚠️ **Next Steps:**\n"
            response += "1. Do not self-diagnose or self-medicate\n"
            response += "2. Note any other symptoms (fever, weight loss, fatigue)\n"
            response += "3. Visit a doctor for proper clinical examination\n\n"
            response += "📸 **For lab reports:** Try taking a clearer, well-lit photo of the report for better analysis."

            return {
                "response": response,
                "details": {
                    "dimensions": f"{width}x{height}",
                    "format": img_format,
                    "type": "symptom_photo_fallback"
                },
                "status": "success"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failure: {str(e)}")

@app.get("/stats/global-hiv")
def get_global_hiv_stats():
    """Returns global HIV statistics for the Visitor section."""
    return {
        "years": ["2021", "2022", "2023", "2024"],
        "new_infections": [1.5, 1.3, 1.3, 1.2],
        "on_art": [28.7, 29.8, 30.6, 31.8],
        "summary": {
            "living_with_hiv": "38M+",
            "on_art_2024": "31.8M",
            "new_infections_2024": "1.2M"
        },
        "status": "success"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
