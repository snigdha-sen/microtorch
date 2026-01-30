import hydra
from omegaconf import DictConfig, OmegaConf

from src.run_fit import run_fit

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Enforce required args
    if cfg.data.image is None:
        raise ValueError("data.image is required")

    print("Running with config:\n")
    print(OmegaConf.to_yaml(cfg))

    run_fit(cfg)

if __name__ == "__main__":
    main()
