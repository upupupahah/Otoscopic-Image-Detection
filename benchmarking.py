import time
import numpy as np
import onnxruntime as ort

def run_benchmark(model_path, device='CPU', batch_size=1, test_duration=5):
    providers = ['CUDAExecutionProvider'] if device == 'GPU' else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    input_shape = [batch_size if dim == 'batch_size' else dim for dim in session.get_inputs()[0].shape]
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # прогрев
    for _ in range(10):
        session.run(None, {input_name: dummy_input})

    frame_count = 0
    start_time = time.time()
    while (time.time() - start_time) < test_duration:
        session.run(None, {input_name: dummy_input})
        frame_count += batch_size

    duration = time.time() - start_time
    fps = frame_count / duration

    print(f"\n{model_path} на {device}:")
    print(f"FPS: {fps:.1f} (за {duration:.1f} сек)")
    print(f"Размер батча: {batch_size}")
    print(f"Всего кадров: {frame_count}")

if __name__ == "__main__":
    # Настройки теста
    MODEL = "model.onnx"
    BATCH_SIZE = 16
    TEST_DURATION = 30  # длительность теста

    # Проверка доступности GPU
    run_benchmark(MODEL, 'GPU', BATCH_SIZE, TEST_DURATION)
    # Тест на CPU
    run_benchmark(MODEL, 'CPU', BATCH_SIZE, TEST_DURATION)