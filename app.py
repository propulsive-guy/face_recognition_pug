from flask import Flask, request, jsonify
from keras_facenet import FaceNet
import numpy as np
import cv2
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import base64
import pickle
import io
from PIL import Image
import os

app = Flask(__name__)

# ------------------- MongoDB Connection -------------------
uri = "mongodb+srv://test1:test121@cluster0.3jrysla.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Test connection
try:
    client.admin.command('ping')
    print("✅ Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

db = client["facenet_db"]
collection = db["faces"]

# ------------------- Initialize FaceNet -------------------
embedder = FaceNet()

# ------------------- Helper Functions -------------------
def read_image(file_bytes):
    """Read and preprocess image bytes"""
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    image = np.array(image)
    return image

def extract_face(img):
    """Crop face using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))  # required size for FaceNet
    return face

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ------------------- 1️⃣ Register Endpoint -------------------
@app.route("/register", methods=["POST"])
def register_person():
    try:
        name = request.form.get("name")
        image_file = request.files.get("image")

        if not name or not image_file:
            return jsonify({"error": "Name and image required"}), 400

        image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # store in MongoDB
        person = {"name": name, "image": encoded_image}
        collection.insert_one(person)

        return jsonify({"message": f"Image stored for {name}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- 2️⃣ Train Endpoint -------------------
    
@app.route("/", methods=["GET"])
def printhealth():
    return "api is live"


@app.route("/train", methods=["POST"])
def train_faces():
    try:
        all_faces = list(collection.find())
        if not all_faces:
            return jsonify({"error": "No images in database"}), 400

        embeddings = []
        labels = []

        for face_doc in all_faces:
            name = face_doc["name"]
            image_data = base64.b64decode(face_doc["image"])
            img = read_image(image_data)
            face = extract_face(img)

            if face is None:
                continue

            face = np.expand_dims(face, axis=0)
            emb = embedder.embeddings(face)
            embeddings.append(emb[0])
            labels.append(name)

        # Save model locally
        with open("embeddings.pkl", "wb") as f:
            pickle.dump({"embeddings": embeddings, "labels": labels}, f)

        return jsonify({"message": f"Trained on {len(labels)} faces"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- 3️⃣ Recognize Endpoint -------------------
@app.route("/recognize", methods=["POST"])
def recognize_person():
    try:
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "Image required"}), 400

        if not os.path.exists("embeddings.pkl"):
            return jsonify({"error": "No trained embeddings found. Train first."}), 400

        # Load trained embeddings
        with open("embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        embeddings = data["embeddings"]
        labels = data["labels"]

        # Read uploaded image
        image_bytes = image_file.read()
        img = read_image(image_bytes)
        face = extract_face(img)

        if face is None:
            return jsonify({"error": "No face detected"}), 400

        face = np.expand_dims(face, axis=0)
        emb = embedder.embeddings(face)[0]

        # Compare with all embeddings
        similarities = [cosine_similarity(emb, e) for e in embeddings]
        max_sim = max(similarities)
        best_match = labels[np.argmax(similarities)]

        THRESHOLD = 0.6  # adjust based on accuracy
        if max_sim >= THRESHOLD:
            return jsonify({
                "name": best_match,
                "similarity": float(max_sim)
            }), 200
        else:
            return jsonify({
                "name": "Unknown",
                "similarity": float(max_sim)
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Run Server -------------------
# ------------------- Run Server -------------------
if __name__ == "__main__":
   # Cloud Run sets this automatically
    app.run(host="0.0.0.0", port=port)
