OUTPUT_PATH="data/dns_noises"
mkdir -p "$OUTPUT_PATH"

BLOB_NAMES=(
    noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2
)

AZURE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset"

for BLOB in "${BLOB_NAMES[@]}"
do
    URL="$AZURE_URL/$BLOB"
    echo "Downloading and extracting: $BLOB"

    curl -s "$URL" | tar -C "$OUTPUT_PATH" -xj

    if [ $? -eq 0 ]; then
        echo "Finished: $BLOB"
    else
        echo "Failed to download or extract: $BLOB"
    fi
done

echo "All files processed. Extracted to: $OUTPUT_PATH"
