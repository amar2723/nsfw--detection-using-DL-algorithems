from flask import Flask, request, jsonify, send_file, render_template
import os
import cv2
import numpy as np
import concurrent.futures
from werkzeug.utils import secure_filename
import torch

# Initialize Flask app
app = Flask(__name__)

# Set upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check for valid video file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Detect NSFW content (assumes 'person' = potential NSFW)
def is_nsfw(frame):
    results = model(frame)
    nsfw_regions = []
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.5 and int(cls) == 0:  # Class 0 = person
            x1, y1, x2, y2 = map(int, xyxy)
            nsfw_regions.append((x1, y1, x2, y2))
    return nsfw_regions

# Apply Gaussian blur to detected regions
def blur_nsfw_content(frame, nsfw_regions):
    for x1, y1, x2, y2 in nsfw_regions:
        region = frame[y1:y2, x1:x2]
        if region.size > 0:
            blurred = cv2.GaussianBlur(region, (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred
    return frame

# Homepage route
@app.route('/')
def home():
    return render_template('index.html')

# Handle video upload and processing
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_path = process_video(filepath)

        return jsonify({'download_url': f'/download/{os.path.basename(processed_path)}'})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

# Download processed video
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

# Process the entire video
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(input_path))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(process_frame, frames))

    for frame in processed_frames:
        out.write(frame)

    cap.release()
    out.release()
    return output_path

# Process a single frame
def process_frame(frame):
    regions = is_nsfw(frame)
    if regions:
        frame = blur_nsfw_content(frame, regions)
    return frame

if __name__ == '__main__':
    app.run(debug=True)
