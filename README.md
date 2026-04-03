# SignSense: Real-Time Sign Language Detection Web App

Flask web interface for your MediaPipe + LSTM sign language detection project.

---

## System Architecture

```mermaid
flowchart TD
    CAM[📷 Webcam]
    MP[MediaPipe Holistic\nmin_conf = 0.5]
    KP[extract_keypoints\npose 132 + lh 63 + rh 63 + face 1404 = 1662]
    BUF[Sequence Buffer\n30 frames rolling]
    LSTM[LSTM Model\naction.h5]
    PRED[Softmax Predictions\nhello · thanks · iloveyou]
    SENT[Sentence Buffer\nmax 5 words · threshold 0.7]
    FLASK[Flask Server]
    MJPEG[/video_feed\nMJPEG Stream/]
    STATE[/state\nJSON Polling 200ms/]
    RESET[/reset\nPOST/]
    UI[Browser UI\nLive feed + Probabilities + Sentence]

    CAM --> MP
    MP -->|landmarks| KP
    KP --> BUF
    BUF -->|len == 30| LSTM
    LSTM --> PRED
    PRED -->|argmax > 0.7| SENT
    SENT --> STATE

    MP -->|styled frame| MJPEG
    FLASK --> MJPEG
    FLASK --> STATE
    FLASK --> RESET

    MJPEG --> UI
    STATE --> UI
    UI -->|clear btn| RESET
```

---

## Data Flow — Keypoint Extraction

```mermaid
flowchart LR
    subgraph Frame
        POSE[Pose\n33 landmarks × 4\n= 132 values]
        LH[Left Hand\n21 landmarks × 3\n= 63 values]
        RH[Right Hand\n21 landmarks × 3\n= 63 values]
        FACE[Face\n468 landmarks × 3\n= 1404 values]
    end

    POSE --> CONCAT
    LH   --> CONCAT
    RH   --> CONCAT
    FACE --> CONCAT
    CONCAT[np.concatenate\n→ 1662-dim vector] --> SEQ[30-frame Sequence\nshape 30 × 1662]
    SEQ --> MODEL[LSTM Model]
```

---

## LSTM Model Architecture

```mermaid
graph TD
    IN[Input\n30 × 1662]
    L1[LSTM 64\nreturn_sequences=True · relu]
    L2[LSTM 128\nreturn_sequences=True · relu]
    L3[LSTM 64\nreturn_sequences=False · relu]
    D1[Dense 64 · relu]
    D2[Dense 32 · relu]
    OUT[Dense 3 · softmax\nhello · thanks · iloveyou]

    IN --> L1 --> L2 --> L3 --> D1 --> D2 --> OUT
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your trained model file (`action.h5`) in the project root.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Project Structure

```
sign_language_app/
├── app.py               # Flask backend + video stream + prediction logic
├── requirements.txt
├── action.h5            # ← put your trained model here
└── templates/
    └── index.html       # Dark neural UI
```

---

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main UI |
| `/video_feed` | GET | MJPEG stream with landmarks |
| `/state` | GET | JSON — predictions, sentence, FPS, confidence |
| `/reset` | POST | Clears sentence + sequence buffer |

---

> If `action.h5` is missing, the app still runs — webcam feed and MediaPipe landmarks are active, predictions are skipped.
