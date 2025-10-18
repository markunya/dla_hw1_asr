import warnings

import hydra
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pathlib import Path

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    text_encoder = instantiate(config.text_encoder)

    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    print(model)

    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        metrics["inference"].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )

    save_path = Path(config.inferencer.save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        logger=logger,
        writer=writer,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
