from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
import uuid
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
PREDICTED_FOLDER = 'static/predicted'
OUTPUT_VIDEO_FOLDER = 'static/output_videos'
MODEL_PATH = 'svm_model.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTED_FOLDER'] = PREDICTED_FOLDER
app.config['OUTPUT_VIDEO_FOLDER'] = OUTPUT_VIDEO_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_VIDEO_FOLDER, exist_ok=True)

# -------------------------------------
# 1️⃣ Load ResNet50 feature extractor
# -------------------------------------
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_cnn_features(frames):
    """Extract 2048-dim CNN features (same as training)."""
    if len(frames) == 0:
        return None

    feats = []
    for frame in frames:
        arr = image.img_to_array(frame)
        arr = np.expand_dims(arr, axis=0)
        arr = tf.keras.applications.resnet50.preprocess_input(arr)
        feat = resnet.predict(arr, verbose=0)
        feats.append(feat.flatten())

    return np.mean(feats, axis=0)


# -------------------------------------
# 2️⃣ Load the trained SVM model
# -------------------------------------
try:
    svm_model = joblib.load(MODEL_PATH)
    print("SVM Model Loaded Successfully!")
except Exception as e:
    print("Error loading SVM model:", e)
    svm_model = None


# -------------------------------------
# 3️⃣ Predict a single frame (5-frame window)
# -------------------------------------
frame_buffer = []


def predict_frame(frame):
    global frame_buffer

    frame_resized = cv2.resize(frame, (224, 224))
    frame_buffer.append(frame_resized)

    if len(frame_buffer) > 5:
        frame_buffer.pop(0)

    if len(frame_buffer) < 5:
        return "Processing..."

    try:
        features = extract_cnn_features(frame_buffer)
        pred = svm_model.predict([features])[0]
        return "Threat" if pred == 1 else "Normal"
    except Exception as e:
        print("Prediction Error:", e)
        return "Prediction Error"


# -------------------------------------
# 4️⃣ Video Processing
# -------------------------------------
def process_video(video_path):
    if svm_model is None:
        return None, None, None, "Error: Model not loaded."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None, "Error: Could not open video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = f"output_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(app.config['OUTPUT_VIDEO_FOLDER'], output_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_predictions = []
    predicted_files = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = predict_frame(frame)
        timestamp = frame_count / fps

        color = (0, 0, 255) if label == "Threat" else (0, 255, 0)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)

        # Save key frames (Threat + Normal)
        if label in ["Threat", "Normal"]:
            frame_filename = f"{label}_{uuid.uuid4().hex}.png"
            frame_path = os.path.join(PREDICTED_FOLDER, frame_filename)
            cv2.imwrite(frame_path, frame)

            frame_predictions.append((frame_count, f"{timestamp:.2f}s", label))
            predicted_files.append((frame_filename, label))

        frame_count += 1

    cap.release()
    out.release()

    return frame_predictions, predicted_files, output_filename, None


# -------------------------------------
# 5️⃣ Flask Routes
# -------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('upload'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('upload'))

    if file and (file.filename.endswith('.mp4') or file.filename.endswith('.avi')):
        filename = f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        results = process_video(filepath)

        frame_predictions, predicted_files, output_video, error = results

        if error:
            return render_template('upload.html', error=error)

        return render_template(
            "result.html",
            video_url=output_video,
            original_video=filename,
            frame_predictions=frame_predictions,
            predicted_files=predicted_files
        )

    return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(debug=True)
