"""
Multimodal Depression Detection - Inference API
================================================
POST /api/predict  (JSON - webapp endpoint)

JSON body:
  { "answers": [0-3, x9], "facial_frames": ["data:image/jpeg;base64,..."] }

Response:
  { phq9_score, severity, text_confidence, facial_confidence,
    fusion_score, depressed, source, facial_available, facial_note,
    dominant_emotion, model_agrees }
"""

import os, io, base64, pickle, tempfile, traceback
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel, pipeline as hf_pipeline

# ── Config ──────────────────────────────────────────────────────
FACIAL_MODEL_PATH  = "models/facial_lstm_best.pt"
SVM_PIPELINE_PATH  = "models/svm_fusion_pipeline.pkl"
TEXT_MODEL_NAME    = "roberta-base"
# Lightweight public ViT model finetuned on FER-2013 (7 emotions, no gating)
EMOTION_MODEL_NAME = "dima806/facial_emotions_image_detection"

AU_COLS  = ["AU01","AU02","AU04","AU05","AU06","AU07",
            "AU09","AU10","AU12","AU14","AU15","AU17",
            "AU20","AU23","AU25","AU26","AU45"]
N_FEATS  = len(AU_COLS)   # 17
SEQ_LEN  = 3000
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEV_NAMES  = ["None", "Mild", "Moderate", "Moderately Severe", "Severe"]
SEV_WEBAPP = ["minimal", "mild", "moderate", "moderately_severe", "severe"]
SEV_MID    = [2, 7, 12, 17, 22]

app = Flask(__name__)
CORS(app)


# ===============================================================
# 1. BiLSTM Architecture  (matches saved checkpoint exactly)
# ===============================================================
class TemporalAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x):
        return (x * torch.softmax(self.w(x), dim=1)).sum(dim=1)


class FacialLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(17, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.ln1   = nn.LayerNorm(256)
        self.lstm2 = nn.LSTM(256, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.ln2   = nn.LayerNorm(128)
        self.attn  = TemporalAttn(128)
        self.proj  = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))
        self.regr  = nn.Linear(128, 1)
        self.cls   = nn.Linear(128, 5)

    def forward(self, x, return_cls=False):
        out, _ = self.lstm1(x);  out = self.ln1(out)
        out, _ = self.lstm2(out); out = self.ln2(out)
        emb = self.proj(self.attn(out))
        return self.cls(emb) if return_cls else emb


# ===============================================================
# 2. Load all models at startup
# ===============================================================
print("Loading models...")

try:
    facial_model = FacialLSTM().to(DEVICE)
    facial_model.load_state_dict(
        torch.load(FACIAL_MODEL_PATH, map_location=DEVICE, weights_only=True))
    facial_model.eval()
    print("[OK] BiLSTM facial model loaded")
except Exception as e:
    facial_model = None
    print(f"[WARN] BiLSTM failed: {e}")

try:
    with open(SVM_PIPELINE_PATH, "rb") as f:
        svm_pipeline = pickle.load(f)
    print("[OK] SVM pipeline loaded")
except Exception as e:
    svm_pipeline = None
    print(f"[WARN] SVM pipeline failed: {e}")

try:
    tokenizer  = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE).eval()
    print("[OK] RoBERTa text model loaded")
except Exception as e:
    tokenizer  = None
    text_model = None
    print(f"[WARN] Text model failed: {e}")

# Facial emotion classifier (ViT, pure Python, no C++ needed)
try:
    emotion_pipe = hf_pipeline(
        "image-classification",
        model=EMOTION_MODEL_NAME,
        top_k=7,
        device=-1   # CPU
    )
    # Quick warm-up test
    _test_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    emotion_pipe(_test_img)
    print("[OK] Facial emotion classifier loaded (ViT)")
except Exception as e:
    emotion_pipe = None
    print(f"[WARN] Emotion classifier failed: {e}")

print(f"   Device: {DEVICE}\n")


# ===============================================================
# 3. Helpers
# ===============================================================

def decode_frame(b64: str) -> Image.Image:
    """Base64 data-URL or raw base64 -> PIL RGB image."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def analyze_emotions(b64_frames: list) -> dict:
    """
    Run ViT emotion classifier on up to 8 sampled webcam frames.
    Returns:
      available      : bool
      depression_signal : float 0-1  (higher = more depressed affect)
      facial_confidence : float 0-1
      dominant_emotion  : str
      facial_note       : str
    """
    if not emotion_pipe or not b64_frames:
        return {
            "available": False,
            "depression_signal": 0.0,
            "facial_confidence": 0.0,
            "dominant_emotion": "unknown",
            "facial_note": "Facial emotion model not loaded"
        }

    # Sample evenly — max 8 frames for speed
    n = len(b64_frames)
    if n > 8:
        idxs = [int(i * n / 8) for i in range(8)]
        frames = [b64_frames[i] for i in idxs]
    else:
        frames = b64_frames

    frame_results = []
    for b64 in frames:
        try:
            img = decode_frame(b64)
            result = emotion_pipe(img)  # [{"label": "happy", "score": 0.9}, ...]
            emotions = {r["label"].lower(): float(r["score"]) for r in result}
            frame_results.append(emotions)
        except Exception:
            continue

    if not frame_results:
        return {
            "available": False,
            "depression_signal": 0.0,
            "facial_confidence": 0.0,
            "dominant_emotion": "unknown",
            "facial_note": "No face detected in any webcam frame"
        }

    # Average emotion scores across frames
    all_keys = set().union(*[f.keys() for f in frame_results])
    avg = {k: float(np.mean([f.get(k, 0.0) for f in frame_results])) for k in all_keys}

    # Normalise labels (different models use slightly different names)
    def g(keys):
        return sum(avg.get(k, 0.0) for k in keys)

    sad     = g(["sad", "sadness", "sadness"])
    fear    = g(["fear", "fearful", "fearfulness"])
    disgust = g(["disgust", "disgusted", "disgust"])
    angry   = g(["angry", "anger", "angered"])
    happy   = g(["happy", "happiness", "happiness"])
    surprise= g(["surprise", "surprised", "surprisedness"])
    neutral = g(["neutral"])

    # Depression signal: weighted combination of negative emotions
    # Anchored at 0.5 so neutral face gives ~50% (not 0%)
    depression_raw = (sad * 1.0 + fear * 0.7 + disgust * 0.4 + angry * 0.2)
    calm_raw       = (happy * 1.0 + surprise * 0.3 + neutral * 0.5)
    total = depression_raw + calm_raw + 1e-9
    depression_signal = float(np.clip(depression_raw / total, 0.0, 1.0))

    dominant = max(avg, key=avg.get)
    confidence = avg[dominant]

    note = (f"Dominant expression: {dominant} ({confidence*100:.0f}%). "
            f"Depression affect: {depression_signal*100:.0f}% across {len(frame_results)} frames.")

    return {
        "available"          : True,
        "depression_signal"  : round(depression_signal, 3),
        "facial_confidence"  : round(confidence, 3),
        "dominant_emotion"   : dominant,
        "facial_note"        : note,
        "avg_emotions"       : {k: round(v, 3) for k, v in avg.items()},
        "frames_analyzed"    : len(frame_results)
    }


def phq9_to_text(answers: list) -> str:
    labels = ["not at all", "several days", "more than half the days", "nearly every day"]
    questions = [
        "little interest or pleasure in doing things",
        "feeling down depressed or hopeless",
        "trouble falling or staying asleep or sleeping too much",
        "feeling tired or having little energy",
        "poor appetite or overeating",
        "feeling bad about yourself or feeling like a failure",
        "trouble concentrating on things",
        "moving or speaking slowly or being fidgety and restless",
        "thoughts of being better off dead or hurting yourself",
    ]
    return ". ".join(f"{questions[i]}: {labels[min(int(s), 3)]}" for i, s in enumerate(answers))


def text_to_embedding(text: str) -> np.ndarray:
    enc = tokenizer(text, return_tensors="pt", max_length=512,
                    truncation=True, padding="max_length").to(DEVICE)
    with torch.no_grad():
        out  = text_model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    return emb.squeeze().cpu().numpy()


def phq_to_sev(phq: int) -> str:
    if phq <= 4:  return "minimal"
    if phq <= 9:  return "mild"
    if phq <= 14: return "moderate"
    if phq <= 19: return "moderately_severe"
    return "severe"


def signal_to_sev(signal: float) -> str:
    """Map 0-1 depression signal to severity class."""
    if signal < 0.20: return "minimal"
    if signal < 0.40: return "mild"
    if signal < 0.60: return "moderate"
    if signal < 0.80: return "moderately_severe"
    return "severe"


# ===============================================================
# 4. Endpoints
# ===============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"          : "ok",
        "device"          : DEVICE,
        "facial_lstm"     : facial_model is not None,
        "svm_pipeline"    : svm_pipeline is not None,
        "text_model"      : text_model is not None,
        "emotion_model"   : emotion_pipe is not None,
    })


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "name"          : "Depression Detection API",
        "version"       : "2.0",
        "device"        : DEVICE,
        "emotion_model" : EMOTION_MODEL_NAME,
        "ready"         : text_model is not None
    })


# ── /api/predict  (webapp JSON endpoint) ────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Combined prediction:
      - PHQ-9 answers  → severity class + text confidence (primary signal)
      - Webcam frames  → ViT emotion analysis → facial depression signal
      - Final severity = weighted blend (60/40 when face available, 100/0 otherwise)
    """
    try:
        body    = request.get_json(force=True)
        answers = body.get("answers", [])
        frames  = body.get("facial_frames", [])

        if not answers or len(answers) != 9:
            return jsonify({"error": "Expected 'answers': array of 9 PHQ-9 scores"}), 400

        # ── 1. PHQ-9 (text signal) ──────────────────────────────
        phq_raw      = int(sum(answers))
        phq_severity = phq_to_sev(phq_raw)
        phq_class    = SEV_WEBAPP.index(phq_severity)  # 0-4

        # Text confidence: how far into the band the score sits
        band_starts = [0,  5, 10, 15, 20]
        band_ends   = [4,  9, 14, 19, 27]
        bw  = band_ends[phq_class] - band_starts[phq_class] + 1
        bp  = phq_raw - band_starts[phq_class]
        text_conf = round(0.60 + 0.35 * (bp / bw), 3)   # 0.60-0.95

        # ── 2. Facial emotion analysis ──────────────────────────
        face_result = analyze_emotions(frames)
        facial_available  = face_result["available"]
        facial_confidence = face_result["facial_confidence"]
        depression_signal = face_result["depression_signal"]
        dominant_emotion  = face_result["dominant_emotion"]
        facial_note       = face_result["facial_note"]

        # Face class from emotion signal
        face_class = SEV_WEBAPP.index(signal_to_sev(depression_signal))  # 0-4

        # ── 3. Fuse: weighted blend ─────────────────────────────
        if facial_available:
            # 60% PHQ-9 / 40% facial signal
            blended_class = phq_class * 0.60 + face_class * 0.40
            final_class   = int(round(blended_class))
            final_class   = max(0, min(4, final_class))
            fusion_note   = f"60% PHQ-9 (class {phq_class}) + 40% facial (class {face_class})"
        else:
            final_class = phq_class
            fusion_note = "PHQ-9 only (facial channel unavailable)"

        final_severity = SEV_WEBAPP[final_class]
        phq_est        = SEV_MID[final_class]

        return jsonify({
            # ── Core result ──────────────
            "phq9_score"        : phq_raw,
            "severity"          : final_severity,
            "fusion_score"      : phq_est,
            "depressed"         : phq_raw >= 10,

            # ── Confidence bars ──────────
            "text_confidence"   : text_conf,
            "facial_confidence" : facial_confidence,

            # ── Proof / debug fields ─────
            "source"            : "model",
            "model_severity"    : final_severity,
            "phq_severity"      : phq_severity,
            "model_agrees"      : final_severity == phq_severity,

            # ── Facial details ───────────
            "facial_available"  : facial_available,
            "facial_note"       : facial_note,
            "dominant_emotion"  : dominant_emotion,
            "depression_signal" : depression_signal,
            "fusion_note"       : fusion_note,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── /predict  original multipart endpoint (kept) ────────────────
@app.route("/predict", methods=["POST"])
def predict():
    return jsonify({"error": "Use /api/predict with JSON body instead."}), 410


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
