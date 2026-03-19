import hydra
from omegaconf import DictConfig, OmegaConf

from src.run_fit import run_fit
from src.utils.plot_results import plot_param_maps

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the microtorch fitting pipeline.
    Expects a Hydra config (DictConfig).
    """

    # Enforce required args
    if cfg.data.image is None:
        raise ValueError("data.image is required")

    print("Running with config:\n")
    print(OmegaConf.to_yaml(cfg))

    _, modelfunc, out_file = run_fit(cfg)

    if cfg.plot.enabled:
        plot_param_maps(out_file, modelfunc, zslice=cfg.plot.zslice)


if __name__ == "__main__":
    main()

