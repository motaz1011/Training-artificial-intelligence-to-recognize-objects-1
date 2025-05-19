from flask import Flask, request, jsonify
from inference import load_model, predict
import os
from flask_cors import CORS # استيراد CORS

app = Flask(__name__)
CORS(app) # تفعيل CORS لجميع المسارات في التطبيق
model = load_model()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    result = predict(image_path, model)

    # عبارات الطباعة للمساعدة في Debugging (يمكنك تركها أو إزالتها بعد حل المشكلة)
    print("Prediction result:", result)
    try:
        json_response = jsonify(result)
        print("JSON response object:", json_response)
        print("JSON response data:", json_response.get_data(as_text=True))
    except Exception as e:
        print("Error during jsonify or getting data:", e)


    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)