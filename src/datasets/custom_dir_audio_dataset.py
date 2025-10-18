from pathlib import Path
import torchaudio  # добавлено

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, target_sr=16000, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix.lower() in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)

                info = torchaudio.info(str(path))
                length = int(info.num_frames * target_sr / info.sample_rate)
                entry["audio_len"] = length

                if transcription_dir and Path(transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()

            if len(entry) > 0:
                data.append(entry)

        super().__init__(data, *args, **kwargs)
