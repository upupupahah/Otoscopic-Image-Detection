from ts.torch_handler.base_handler import BaseHandler
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import json
import logging
import os

logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.transform = None
        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, context):
        try:
            # 1. Проверка наличия файлов
            model_dir = context.system_properties.get("model_dir")
            model_path = os.path.join(model_dir, context.manifest["model"]["serializedFile"])

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # 2. Загрузка классов
            class_file = os.path.join(model_dir, "class_mapping.json")
            if not os.path.exists(class_file):
                raise FileNotFoundError(f"Class mapping file not found at {class_file}")

            with open(class_file) as f:
                self.class_names = json.load(f)

            # 3. Инициализация модели ТОЧНО как при обучении
            self.model = models.efficientnet_b4(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, len(self.class_names))

            # 4. Загрузка весов с проверкой
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device).eval()

            # 5. Преобразования изображений
            self.transform = transforms.Compose([
                transforms.Resize(380),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"Model init error: {str(e)}") from e

    def preprocess(self, data):
        try:
            image = data[0].get("data") or data[0].get("body")
            image = Image.open(io.BytesIO(image)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0)
            return tensor.to(self.device)
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def inference(self, data):
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, output):
        try:
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, indices = torch.topk(probs, 1)  # Берем только топ-1

            batch_results = []
            for batch_idx in range(output.size(0)):
                class_idx = str(indices[batch_idx][0].item())
                batch_results.append([{
                    "class": self.class_names[class_idx],
                    "confidence": round(conf[batch_idx][0].item(), 4)
                }])

            return batch_results

        except Exception as e:
            logger.error(f"Postprocess error: {str(e)}")
            return [[{"error": "Prediction failed"}]]
