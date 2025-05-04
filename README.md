# Otoscopic Image Detection


## Отчет по заданию 3
---

> **upd**: Я сильно сомневаюсь в правильности моей реализации
> Соблюдены не все условия задания (см. последний пункт отчета)


> **upd 2**: Файлы модели и mar архив не помещаются на гитхаб, вместо них в папке project `txt` файлы. Про создание `mar` написано ниже, а файл `model.pth` нужно поместить в корень проекта вместе с `handler.py`


### Запуск проекта

#### Подготовка модели
Перед запуском необходимо создать `.mar` архив модели. Операция производится единожды:

```bash
torch-model-archiver --model-name my_model --version 1.0 --serialized-file model.pth --handler handler.py --extra-files "class_mapping.json" --export-path there_is_our_server --force
```

#### Запуск torchserve
```bash
torchserve --start --model-store there_is_our_server --models my_model=my_model.mar --ncs --disable-token-auth
```

#### Запуск flask-api сервера
```bash
python3 server.py
```

**Или**:
Запуск файла `server.py`

#### Обращение к модели
Есть 2 способа:

- Напрямую через `torchserve`:
```bash
curl -X POST http://localhost:8080/predictions/my_model -H "Content-Type: image/jpeg" --data-binary "@path.jpg
```
- Через `flask-api`:
```bash
curl -X POST -F "image=@dataset/test/1/aom (511).jpg" http://localhost:5000/predict
```

#### Остановка torchserve
```bash
torchserve --stop
```

### Проблемы реализации
- Не реализована *Swagger документация* по причине отсутствия знаний в этой области (я вообще не знаю, что это)
- Не реализован *Healthcheck сервиса* - по той же причине
- Нет понимания, зачем в таком небольшом проекте необходим `torchserve`, который создает лишний геморрой, при этом не давая никаких преимуществ (те же функции можно реализовать сильно проще и быстрее, используя только `flask` и `torch`, как я писал выше - через функцию распознавания одного изображения)

