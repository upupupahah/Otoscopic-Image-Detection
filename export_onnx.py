import torch
from torchvision import models
import torch.nn as nn

model_path = "model.pth"
classes = ['1', '2', '3', '4', '5']
onnx_path = "model.onnx"
size = 380

model = models.efficientnet_b4(weights=None)
model.classifier[1] = nn.Linear(
    in_features=model.classifier[1].in_features,
    out_features=len(classes)
)
model.load_state_dict(torch.load(model_path, map_location="cuda"))
model.eval()

dummy_input = torch.randn(1, 3, size, size)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print(f"ONNX модель сохранена: ", onnx_path)