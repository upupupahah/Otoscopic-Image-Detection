from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantType, QuantFormat
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import onnx
import time
import sys

# Конфигурация
onnx_path = "model.onnx"
quant_path = "model_quantized.onnx"
calibration_dir = "dataset/train"
batch_size = 4 # больше 4 крашило систему

calibration_transform = transforms.Compose([
    transforms.Resize(380),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CustomDataReader(CalibrationDataReader):
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
        self.total_batches = len(data_loader)
        self.current_batch = 0
        self.start_time = time.time()

        print("\nстарт калибровки")
        print(f"всего батчей: {self.total_batches}")
        self._print_progress(0)

    def _print_progress(self, progress):
        bars = int(progress * 20)
        percent = int(progress * 100)
        elapsed = int(time.time() - self.start_time)
        sys.stderr.write(
            f"\r[{'█' * bars}{' ' * (20 - bars)}] {percent}% "
            f"({self.current_batch}/{self.total_batches}) "
            f"прошло: {elapsed}s"
        )
        sys.stderr.flush()

    def get_next(self) -> dict:
        try:
            images, _ = next(self.iter)
            self.current_batch += 1
            progress = self.current_batch / self.total_batches
            self._print_progress(progress)
            return {"input": images.numpy()}
        except StopIteration:
            total_time = int(time.time() - self.start_time)
            sys.stderr.write(f"\nкалибровка завершена за {total_time} секунд\n")
            return None

def quantize_efficientnet():
    print("загрузка модели...")
    model = onnx.load(onnx_path)
    print(f"размер батча для калибровки: {batch_size}")
    print("\nначало квантизации...")
    start_time = time.time()

    quantize_static(
        model_input=onnx_path,
        model_output=quant_path,
        calibration_data_reader=CustomDataReader(
            DataLoader(
                datasets.ImageFolder(calibration_dir, calibration_transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        ),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        op_types_to_quantize=[
            'Conv', 'MatMul', 'Add', 'Mul', 'Relu', 'Clip',
            'Sigmoid', 'GlobalAveragePool', 'ReduceMean'
        ],
        extra_options={
            'EnableSubgraph': True,
            'ForceQuantizeNoInputCheck': True,
            'MatMulConstBOnly': False
        }
    )

    total_time = int(time.time() - start_time)
    print(f"\nзавершено за {total_time} секунд")
    print(f"Сохранена модель: {quant_path}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    quantize_efficientnet()

    print("\nпроверка квантизации:")
    model = onnx.load(quant_path)
    quant_nodes = sum(1 for node in model.graph.node if node.op_type in ['QuantizeLinear', 'DequantizeLinear'])
    print(f"квантизированные узлы: {quant_nodes}")