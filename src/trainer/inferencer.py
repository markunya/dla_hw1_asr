from pathlib import Path
import csv
import torch
import kenlm
from tqdm.auto import tqdm
from src.trainer.base_trainer import BaseTrainer
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.utils.beam_search import beam_search_decode_lp

class Inferencer(BaseTrainer):
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        logger,
        writer,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        **kwargs
    ):
        super().__init__(
            model=model,
            criterion=None,
            metrics=metrics,
            optimizer=None,
            lr_scheduler=None,
            text_encoder=text_encoder,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            **kwargs
        )

        assert (skip_model_load or config.inferencer.get("from_pretrained") is not None), \
            "Provide checkpoint or set skip_model_load=True"
        
        self.decode_cfg = getattr(self.config, "decoder", {})
        self.use_beam = self.decode_cfg.get("type", "argmax") == "beam"
        self.beam_size = int(self.decode_cfg.get("beam_size", 20))
        self.alpha = float(self.decode_cfg.get("alpha", 0.7))
        self.beta = float(self.decode_cfg.get("beta", 1.5))
        self.prune_topk = self.decode_cfg.get("prune_topk", 50)
        self.lm_path = self.decode_cfg.get("lm_path", None)

        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.text_encoder = text_encoder
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        self.lm = None
        if self.use_beam and self.lm_path:
            try:
                self.lm = kenlm.LanguageModel(self.lm_path)
                tqdm.write(f"[Trainer] Loaded KenLM: {self.lm_path}")
            except Exception as e:
                tqdm.write(f"[Trainer] Failed to load LM ({self.lm_path}): {e}\nContinue without LM.")
                self.lm = None

    @torch.no_grad()
    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if self.use_beam:
            preds = beam_search_decode_lp(
                log_probs=batch["log_probs"],
                log_probs_length=batch["log_probs_length"],
                ind2char=self.text_encoder.ind2char,
                blank_id=self.text_encoder.blank_id,
                beam_size=self.beam_size,
                alpha=self.alpha,
                beta=self.beta,
                lm=self.lm,
                prune_topk=self.prune_topk,
            )
        else:
            argmax_inds = batch["log_probs"].argmax(-1).cpu().numpy()
            lens = batch["log_probs_length"].cpu().numpy()
            preds = [
                self.text_encoder.ctc_decode(argmax_inds[i, : int(lens[i])])
                for i in range(argmax_inds.shape[0])
            ]

        targets = [self.text_encoder.normalize_text(t) for t in batch["text"]]
        audio_paths = batch.get("audio_path", [""] * len(targets))

        if metrics is not None and self.metrics is not None:
            for m in self.metrics["inference"]:
                metrics.update(m.name, m(**batch))

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
            csv_path = self.save_path / part / "predictions.csv"
            header = ["file", "target", "prediction", "wer", "cer"]

            file_exists = csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(header)
                for tgt, pred, ap in zip(targets, preds, audio_paths):
                    w.writerow([
                        Path(ap).name if ap else "",
                        tgt,
                        pred,
                        f"{calc_wer(tgt, pred)*100:.4f}",
                        f"{calc_cer(tgt, pred)*100:.4f}",
                    ])

        batch["pred_text"] = preds
        batch["target_text"] = targets
        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return {} if self.evaluation_metrics is None else self.evaluation_metrics.result()
