#!/bin/bash
set -e  # прерывать при ошибке

# === Настройки ===
ROOT_PATH=$(pwd)
DATA_DIR="$ROOT_PATH/data/datasets/librispeech"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# === Ссылки на части датасета ===
declare -A URLS=(
  ["train-clean-100"]="https://www.openslr.org/resources/12/train-clean-100.tar.gz"
  ["test-clean"]="https://www.openslr.org/resources/12/test-clean.tar.gz"
  ["test-other"]="https://www.openslr.org/resources/12/test-other.tar.gz"
)

# === Скачивание и распаковка ===
for part in "${!URLS[@]}"; do
  url="${URLS[$part]}"
  archive="${part}.tar.gz"

  echo "🔽 Скачиваю $part ..."
  if [ ! -f "$archive" ]; then
    wget -q "$url" -O "$archive"
  else
    echo "✅ Архив уже скачан: $archive"
  fi

  echo "📦 Распаковываю $archive ..."
  tar -xzf "$archive"

  echo "✅ Готово: $part"
done

echo "🎉 Все части LibriSpeech успешно скачаны в: $DATA_DIR"

