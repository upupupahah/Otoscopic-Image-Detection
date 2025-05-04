from flask import Flask, request, jsonify
from handler import ModelHandler
from PIL import Image
import io
import logging

app = Flask(__name__)
model = ModelHandler()


class MockContext:
    def __init__(self):
        self.system_properties = {
            "model_dir": "./",  # Путь к папке с model.pth и class_mapping.json
            "gpu_id": 0
        }
        self.manifest = {
            "model": {
                "serializedFile": "model.pth"  # Имя вашего файла с весами
            }
        }


# Инициализация модели
try:
    context = MockContext()
    model.initialize(context)
    logging.info("Model initialized successfully")
except Exception as e:
    logging.error(f"Model initialization failed: {str(e)}")
    raise


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "working"})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Send image file"}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({"error": "Empty file"}), 400

    try:
        img_bytes = file.read()
        tensor = model.preprocess([{"body": img_bytes}])
        output = model.inference(tensor)
        result = model.postprocess(output)[0][0]
        return jsonify(result)
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": "Processing error"}), 500


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000)