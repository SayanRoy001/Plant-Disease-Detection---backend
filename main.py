from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import torchvision.transforms as transforms
import io
import json
import os
import torch
import timm
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
model = None  
MODEL_PATH = "plant_disease_model.pth"
@app.route("/")
def home():
   return "server ready!"

@app.route('/health')
def health():
    """Lightweight health check (no model load)."""
    return jsonify(status="ok"), 200
def load_model():
    """Load model only once (prevents high memory usage)."""
    global model
    if model is None:  # Load model only if it's not already loaded
        print("🔹 Loading Model...")
        model = timm.create_model('convmixer_1024_20_ks9_p14.in1k', pretrained=True, num_classes=38)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        model.to(torch.device('cpu'))  # Ensure it's on CPU
    return model
#model=load_model()
#print("model loaded")
def load_class_indices(file_path="class_indices.json"):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist!")
        return None
    try:
        with open(file_path, 'r') as f:
            class_indices = json.load(f)
        return class_indices
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return None
    
class_indices=load_class_indices()
class_labels = {int(k): v for k, v in class_indices.items()}

# Preprocessing function (adjust based on your model's input size)
def preprocess_image(img,target_size=(256,256)):
    #img = Image.open(image_file)
    # Convert the image to RGB if it's not already
    img = img.convert('RGB')
    # Resize the image to the target size
    img = img.resize(target_size)
    # Add batch dimension (making it [1, height, width, channels])
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    tensor_img = transform(img)
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img
    

@app.route('/predict', methods=['POST'])
def predict():
   
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    processed_img = processed_image.to('cpu')
    model=load_model()
    with torch.no_grad():
        outputs = model(processed_img)
    predicted_class_index = outputs.argmax(dim=1).item()
    predicted_class_name = class_labels[predicted_class_index]
    print(predicted_class_name)
    return jsonify({"prediction": predicted_class_name })

import google.generativeai as genai
load_dotenv()  
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
# Allow choosing model via env; default to widely available model
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
genai.configure(api_key=GENAI_API_KEY)
from functools import lru_cache

def get_disease_details(disease_name):
    """Generates a detailed explanation, symptoms, and treatment for a given disease using Gemini AI."""
    try:
        model = genai.GenerativeModel(GENAI_MODEL)
        prompt = f"""
        You are a plant pathology assistant. Write a concise, structured note for the disease: {disease_name}.
        Use EXACT section headings with colons so the client can parse them:
        What it is:
        Causes:
        Symptoms:
        Effects on the plant:
        Treatment Plan:
        Prevention Methods:

        Keep sentences simple and to the point. Avoid markdown lists with leading asterisks.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

@lru_cache(maxsize=256)
def cached_disease_details(disease_name: str):
    """Cache wrapper to reduce repeated LLM calls for the same disease."""
    return get_disease_details(disease_name)

@app.route("/diseaseinfo", methods=["GET"])
def get_disease_info():
    """API endpoint to fetch disease information."""
    disease_name = request.args.get('prediction')
    if not disease_name:
        print("⚠️ No disease_name received in request!")
        
    print(disease_name)
    # Use cached variant to avoid excessive token usage & latency.
    disease_info = cached_disease_details(disease_name)
    print(disease_info)
    return jsonify({"info":disease_info})

if __name__ == '__main__':
    # Allow overriding port and debug mode from environment variables.
    # Use 8080 as default (common for container platforms like Cloud Run) if PORT not set.
    port = int(os.getenv('PORT', os.getenv('FLASK_RUN_PORT', '8080')))
    debug_flag = os.getenv('FLASK_DEBUG', '').lower() in ('1', 'true', 'yes')
    # Optional warm model load for platforms where first-request latency matters
    if os.getenv('WARM_MODEL', '').lower() in ('1', 'true', 'yes'):
        try:
            load_model()
            print('Model preloaded at startup.')
        except Exception as e:
            print(f'Warm model load failed: {e}')
    app.run(host='0.0.0.0', port=port, debug=debug_flag)


