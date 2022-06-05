import hydra
import tensorflow as tf
import tensorflow_datasets as tfds
from ktk.utils import tf_set_memory_growth
from omegaconf import DictConfig
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
    dataset = dataset.map(
        preprocess_dataset(mode, is_training), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@hydra.main(version_base="1.2", config_path="./config", config_name="tflite_converter")
def tflite_converter(cfg: DictConfig) -> None:
    tf_set_memory_growth()

    # Load a model to be evaluated and compile it if needed
    converter = None
    model = keras.models.load_model(cfg.input)
    if ".h5" in cfg.input:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(cfg.input)
    assert converter is not None

    # Available operations
    converter.allow_custom_ops = cfg.allow_custom_ops
    if cfg.enable_tf_ops:
        converter.target_spec.supported_ops.add(tf.lite.OpsSet.SELECT_TF_OPS)

    # Post-training quantization
    if cfg.ptq.dynamic_range:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore
    if cfg.ptq.float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore
        converter.target_spec.supported_types = [tf.float16]
    if cfg.ptq.full_integer:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # type: ignore

        data_dir = "~/tensorflow_datasets"
        if cfg.ptq.representative_dataset.data_dir:
            data_dir = cfg.ptq.representative_dataset.data_dir

        ds = tfds.load(
            name=cfg.ptq.representative_dataset.name,
            split=list(cfg.ptq.representative_dataset.split),
            data_dir=data_dir,
            as_supervised=True,
        )[0]

        ds = prepare_dataset(ds, mode=cfg.ptq.representative_dataset.pp_mode, is_training=False)

        def representative_dataset():
            for data in ds.take(100).as_numpy_iterator():
                yield [data[0]]

        converter.representative_dataset = representative_dataset  # type: ignore

    tflite_model = converter.convert()

    open(cfg.output, "wb").write(tflite_model)


if __name__ == "__main__":
    tflite_converter()
