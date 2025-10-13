set -e

LM_DIR="data/lm"
LM_URL="https://us.openslr.org/resources/11/4-gram.arpa.gz"
LM_FILE="$LM_DIR/4-gram.arpa.gz"
LM_ARPA="$LM_DIR/4-gram.arpa"

mkdir -p "$LM_DIR"

echo "📦 Скачиваю языковую модель (LibriSpeech 4-gram)..."
wget --no-check-certificate -O "$LM_FILE" "$LM_URL"

echo "🗜️ Распаковываю..."
gunzip -f "$LM_FILE"

if [ -f "$LM_ARPA" ]; then
    echo "✅ Модель успешно загружена: $LM_ARPA"
else
    echo "❌ Ошибка: файл $LM_ARPA не найден!"
    exit 1
fi

echo ""
echo "🔍 Проверка (Python):"
echo "----------------------------------------"
echo "import kenlm"
echo "lm = kenlm.LanguageModel('$LM_ARPA')"
echo "print(lm.score('the cat sat on the mat', bos=True, eos=True))"
