# Automatic Speech Recognition (ASR)

Этот проект реализует систему **автоматического распознавания речи (ASR)** на основе архитектуры **Conformer + CTC Loss**.
Модель обучалась на подмножестве **LibriSpeech (train-clean-100)**.

---

## 📦 Установка

```bash
git clone https://github.com/markunya/dla_hw1_asr.git
cd dla_hw1_asr
conda create -n dla_hw1 python=3.12.11
conda activate dla_hw1
pip install -r requirements.txt
```

---

## 🔊 Загрузка данных для обучения

Для воспроизведения обучения необходимо скачать шумы и импульсные характеристики,
использованные для аугментации данных:

```bash
bash scripts/download_noises.sh
bash scripts/download_irs.sh
```

Оба скрипта не требуют аргументов и автоматически сохраняют данные в нужные директории.

---

## 🧠 Загрузка моделей
Для загрузки весов используйте скрипт:
```bash
curl -Lo "data/weights/model_best.pth" "$(
  curl -sG --data-urlencode "public_key=https://yadi.sk/d/dT-0ta5WHuKO_Q" \
    "https://cloud-api.yandex.net/v1/disk/public/resources/download" \
  | jq -r .href
)"
```

Для инференса с использованием KenLM (посимвольной LM) можно скачать готовую модель:

```bash
curl -Lo "data/lm/char_lm.arpa" "$(
  curl -sG --data-urlencode "public_key=https://yadi.sk/d/j2fgL98mMoIc3w" \
    "https://cloud-api.yandex.net/v1/disk/public/resources/download" \
  | jq -r .href
)"
```

---

## 🚀 Инференс

Пример запуска инференса с разными вариантами декодирования:

### 1️⃣ Простой argmax (без LM)

```bash
python inference.py inferencer.from_pretrained="data/weights/model_best.pth"
```

### 2️⃣ Beam search без LM

```bash
python inference.py +decoder.type=beam inferencer.from_pretrained="data/weights/model_best.pth"
```

### 3️⃣ Beam search с LM

```bash
python inference.py +decoder.type=beam +decoder.lm_path=data/lm/char_lm.arpa inferencer.from_pretrained="data/weights/model_best.pth"
```
---

## 📘 Demo Notebook

Файл `demo.ipynb` показывает пример:

* как загрузить модель и LM,
* как провести инференс на тестовом наборе,
* и как ввести собственный датасет через Google Drive ссылку.
