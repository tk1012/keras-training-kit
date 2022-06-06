import hydra
import tensorflow as tf
import tensorflow_datasets as tfds
from ktk.utils import tf_set_memory_growth
from omegaconf import DictConfig
from tensorflow import keras


# TODO modularize preprocessing
def preprocess_dataset(shape, mode='tf', is_training=True):
    def _pp(image, label):
        if is_training:
            image = tf.image.resize(image, shape)
            image = tf.cast(image, tf.float32)
            image = keras.applications.imagenet_utils.preprocess_input(
                image, data_format=None, mode=mode
            )
            # TODO: augmentation
        else:
            image = tf.image.resize(image, shape)
            image = tf.cast(image, tf.float32)
            image = keras.applications.imagenet_utils.preprocess_input(
                image, data_format=None, mode=mode
            )
        return image, label

    return _pp


def prepare_dataset(dataset, shape, mode='tf', batch_size=32, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(shape, mode, is_training), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@hydra.main(version_base="1.2", config_path="./config", config_name="evaluate")
def evaluate(cfg: DictConfig) -> None:
    tf_set_memory_growth()

    # Step1: Load a trained model
    model = tf.keras.models.load_model(cfg.evaluate.model_path)

    # Step2: Load a dataset
    data_dir = "~/tensorflow_datasets"
    if cfg.dataset.data_dir:
        data_dir = cfg.dataset.data_dir

    eval_dataset = tfds.load(
        cfg.dataset.name, split=list(cfg.dataset.split), data_dir=data_dir, as_supervised=True
    )[0]

    # TODO: handle multiple inputs
    image_shape = model.inputs[0].shape[1:3]
    eval_dataset = prepare_dataset(eval_dataset, shape=image_shape, mode=cfg.evaluate.pp_mode, is_training=False)

    # Step3: evaluate
    model.evaluate(eval_dataset)


if __name__ == "__main__":
    evaluate()
