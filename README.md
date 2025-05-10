# Отчет о выполнении задания 5

## 1. КОНФИГУРАЦИЯ ИСПОЛЬЗУЕМОЙ СИСТЕМЫ
- **ОС:** Ubuntu 24.04 LTS
- **GPU:** NVIDIA GeForce RTX 2080 Super (8 gb VRAM)
- **CPU:** Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz
- **RAM:** 16 gb DDR3

- **CUDA Version:** 12.8
- **Driver Version:** 570.124.06
- **cuDNN Version:** 9.10.0
- **Python Version:** 3.12.3
- **pycharm-community Version:** 2025.1

## 2. Ознакомление с архитектурой моделей
```bash
netron model.pth
```
```bash
netron model.onnx
```
```bash
netron model_quantized.onnx
```

## 3. Результаты бенчмарка конвертированной модели
|  Устройство  |      FPS      |
|:------------:|:-------------:|
|     GPU      |     204.1     |
|    CPU    |      4.2      |


## 4. Возникшие проблемы
- Не удалось успешно квантизировать модель
- У квантизированной модели скорость ниже, чем у оригинальной `.onnx`

*(остальные отчеты в папке архив)*