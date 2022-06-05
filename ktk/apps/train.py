import hydra
import ktk.models
import tensorflow as tf
import tensorflow_datasets as tfds
from ktk.utils import tf_set_memory_growth
from omegaconf import DictConfig, OmegaConf
from tensorflow import keras


# TODO modularize preprocessing
def preprocess_dataset(mode='tf', is_training=True):
    def _pp(image, label):
        if is_training:
            image = tf.cast(image, tf.float32)
            image = keras.applications.imagenet_utils.preprocess_input(
                image, data_format=None, mode=mode
            )
            # TODO: augmentation
        else:
            image = tf.cast(image, tf.float32)
            image = keras.applications.imagenet_utils.preprocess_input(
                image, data_format=None, mode=mode
            )
        return image, label

    return _pp


def prepare_dataset(dataset, mode='tf', batch_size=32, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(mode, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def train(cfg: DictConfig) -> None:
    tf_set_memory_growth()

    # Step1: Build a model to be trained
    model_def = ktk.models.get(cfg.model.id)
    hparam = model_def.get_default_param()
    hparam.__dict__.update(OmegaConf.to_container(cfg.model.hyper_param))  # type: ignore

    model = model_def.build(**hparam.__dict__)

    model.compile(**cfg.model.compile_param)
    model.summary()

    # Step2: Load a dataset
    data_dir = "~/tensorflow_datasets"
    if cfg.dataset.data_dir:
        data_dir = cfg.dataset.data_dir

    train_dataset, val_dataset = tfds.load(
        cfg.dataset.name, split=list(cfg.dataset.split), data_dir=data_dir, as_supervised=True
    )

    train_dataset = prepare_dataset(train_dataset, mode=cfg.train.pp_mode, is_training=True)
    val_dataset = prepare_dataset(val_dataset, mode=cfg.train.pp_mode, is_training=False)

    # Step3: Train
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="tb_logs/", histogram_freq=1)
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=cfg.train.epoch,
        callbacks=[tensorboard_callback],
    )

    model.save(cfg.train.output_path)


if __name__ == "__main__":
    train()
