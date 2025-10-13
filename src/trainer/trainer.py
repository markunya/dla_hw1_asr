from pathlib import Path

import os
import torch
import pandas as pd
import kenlm
from tqdm import tqdm
from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer
from src.utils.beam_search import beam_search_decode_lp

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, config, model, criterion, metrics, optimizer=None, lr_scheduler=None, **kwargs):
        super().__init__(config, model, criterion, metrics, optimizer, lr_scheduler, **kwargs)

        self.decode_cfg = getattr(self.config, "decoder", {})
        self.use_beam = self.decode_cfg.get("type", "argmax") == "beam"
        self.beam_size = int(self.decode_cfg.get("beam_size", 20))
        self.alpha = float(self.decode_cfg.get("alpha", 0.7))
        self.beta = float(self.decode_cfg.get("beta", 1.5))
        self.prune_topk = self.decode_cfg.get("prune_topk", 50)
        self.lm_path = self.decode_cfg.get("lm_path", None)

        self.lm = None
        if self.use_beam and self.lm_path:
            try:
                self.lm = kenlm.LanguageModel(self.lm_path)
                tqdm.write(f"[Trainer] Loaded KenLM: {self.lm_path}")
            except Exception as e:
                tqdm.write(f"[Trainer] Failed to load LM ({self.lm_path}): {e}\nContinue without LM.")
                self.lm = None

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)

        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            if self.detect_anomaly:
                with torch.autograd.set_detect_anomaly(True):
                    batch["loss"].backward()
            else:
                batch["loss"].backward()

            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """

        if mode == "train":
            self.log_input(**batch)
        else:
            self.log_input(**batch)
            self.log_input(**batch)

    def log_input(self, spectrogram, audio, audio_path, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(os.path.basename(audio_path[0]), image)

        self.writer.add_audio(os.path.basename(audio_path[0]), audio[0], self.sample_rate)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        tqdm.write(str(argmax_inds))

        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        beam_texts = None
        if self.use_beam:
            try:
                beam_texts = beam_search_decode_lp(
                    log_probs=log_probs,
                    log_probs_length=log_probs_length,
                    ind2char=self.text_encoder.ind2char,
                    blank_id=self.text_encoder.blank_id,
                    beam_size=self.beam_size,
                    alpha=self.alpha,
                    beta=self.beta,
                    lm=self.lm,
                    prune_topk=self.prune_topk,
                )
            except Exception as e:
                tqdm.write(f"[Trainer] Beam search упал: {e}. Фоллбек на argmax.")
                beam_texts = None

        final_preds = beam_texts if beam_texts is not None else argmax_texts

        tuples = list(zip(final_preds, text, argmax_texts_raw, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
