"""

"""


from nerf_experiments.source.config.config_utils import load_yaml_config, parse_nerf_config
from nerf_experiments.source.config.runtime_config import to_runtime_train_config
from nerf_experiments.source.train.trainer import Trainer

y = load_yaml_config("config.yaml")
nerf_cfg = parse_nerf_config(y)

# Optional CLI overrides (dot-paths)
cli = {
    # "train.device": "cuda:1",
    # "render.Nc": 96,
    # "train.max_steps": "300k",
    # "seed": 1234,
}

rt_cfg = to_runtime_train_config(
    nerf_cfg,
    cli_overrides=cli,
    save_dir=nerf_cfg.train.out_dir,   # write resolved_config.yaml + input_config.yaml
    original_yaml=y,
)

trainer = Trainer(rt_cfg)
trainer.train()
