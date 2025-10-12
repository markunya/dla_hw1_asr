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
    specs = [
        (it["spectrogram"].squeeze(0) if it["spectrogram"].dim() == 3 else it["spectrogram"])
        for it in dataset_items
    ]
    spec_lens = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    n_mels = specs[0].shape[0]
    T_max = int(spec_lens.max().item())
    padded_specs = torch.zeros(len(specs), n_mels, T_max, dtype=torch.float32)
    for i, s in enumerate(specs):
        padded_specs[i, :, : s.shape[-1]] = s

    waves = [it["audio"] for it in dataset_items]
    wave_lens = torch.tensor([w.shape[-1] for w in waves], dtype=torch.long)
    S_max = int(wave_lens.max().item())
    padded_waves = torch.zeros(len(waves), 1, S_max, dtype=torch.float32)
    for i, w in enumerate(waves):
        padded_waves[i, :, : w.shape[-1]] = w

    text_tensors = [it["text_encoded"].squeeze(0).to(torch.long) for it in dataset_items]
    text_lens = torch.tensor([t.size(0) for t in text_tensors], dtype=torch.long)
    padded_texts = pad_sequence(text_tensors, batch_first=True, padding_value=0)

    texts = [it["text"] for it in dataset_items]
    audio_paths = [it["audio_path"] for it in dataset_items]

    batch = {
        "audio": padded_waves,
        "audio_length": wave_lens,
        "spectrogram": padded_specs,
        "spectrogram_length": spec_lens,
        "text_encoded": padded_texts,
        "text_encoded_length": text_lens,
        "text": texts,
        "audio_path": audio_paths,
    }
    return batch
