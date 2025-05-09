import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import efficientnet_b4
from onnxruntime.quantization import quantize_dynamic


dataset_path = "dataset/test"
model_path = "model.pth"
device = torch.device("cuda")
classes = datasets.ImageFolder(dataset_path).classes
print(f"Определены классы: {len(classes)} — {classes}")

onnx_name = "model.onnx"
quantized_name = "quantized_model.onnx"


# загрузка модели
model = efficientnet_b4(weights=None)
model.classifier[1] = nn.Linear(
    in_features=model.classifier[1].in_features,
    out_features=len(classes)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# экспорт в .onnx
dummy_input = torch.randn(1, 3, 380, 380).to(device)
torch.onnx.export(
    model,
    dummy_input,
    onnx_name,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"}
    }
)

quantize_dynamic(onnx_name, quantized_name)

print("Модели сохранены")
print("В формате .onnx:", onnx_name)
print("Квантизированная:", quantized_name)