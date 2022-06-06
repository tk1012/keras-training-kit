import hydra
import tensorflow as tf
from ktk.utils import tf_set_memory_growth
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.2", config_path="./config", config_name="save_keras_apps")
def save_keras_apps(cfg: DictConfig) -> None:
    tf_set_memory_growth()

    # Step1: Load a trained model
    cls = getattr(tf.keras.applications, cfg.model.id)
    model = cls()
    model.trainable = False

    # Step2: build a model
    model.compile(**OmegaConf.to_container(cfg.model.compile_param))
    model.summary()

    # Step3: evaluate
    model.save(cfg.output_path)


if __name__ == "__main__":
    save_keras_apps()
