import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    spectrograms = [item["spectrogram"].squeeze(0) if item["spectrogram"].dim() == 3 else item["spectrogram"]
                    for item in dataset_items]
    spectrogram_lengths = torch.tensor([spec.shape[-1] for spec in spectrograms], dtype=torch.long)

    max_T = max(s.shape[-1] for s in spectrograms)
    n_mels = spectrograms[0].shape[0]
    padded_spectrograms = torch.zeros(len(spectrograms), n_mels, max_T, dtype=torch.float32)
    for i, s in enumerate(spectrograms):
        padded_spectrograms[i, :, : s.shape[-1]] = s

    text_encoded = [torch.tensor(item["text_encoded"], dtype=torch.long) for item in dataset_items]
    text_encoded_length = torch.tensor([t.size(0) for t in text_encoded], dtype=torch.long)
    padded_text_encoded = pad_sequence(text_encoded, batch_first=True, padding_value=0)

    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    batch = {
        "spectrogram": padded_spectrograms,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": padded_text_encoded,
        "text_encoded_length": text_encoded_length,
        "text": texts,
        "audio_path": audio_paths,
    }

    return batch
