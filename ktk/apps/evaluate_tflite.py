import hydra
import tensorflow as tf
import tensorflow_datasets as tfds
from ktk.utils import tf_set_memory_growth
from omegaconf import DictConfig
from tensorflow import keras


# TODO modularize preprocessing
def preprocess_dataset(mode="tf", is_training=True):
    def _pp(image, label):
        if is_training:
            image = tf.cast(image, tf.float32)
            image = keras.applications.imagenet_utils.preprocess_input(
                image, data_format=None, mode="tf"
            )
            # TODO: augmentation
        else:
            image = tf.cast(image, tf.float32)
            image = keras.applications.imagenet_utils.preprocess_input(
                image, data_format=None, mode="tf"
            )
        return image, label

    return _pp


def prepare_dataset(dataset, mode="tf", batch_size=32, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(
        preprocess_dataset(mode, is_training), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@hydra.main(version_base="1.2", config_path="./config", config_name="evaluate_tflite")
def evaluate_tflite(cfg: DictConfig) -> None:
    tf_set_memory_growth()

    # Step1: Load a tflite model
    interpreter = tf.lite.Interpreter(model_path=cfg.evaluate_tflite.model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Step2: Load a dataset
    data_dir = "~/tensorflow_datasets"
    if cfg.dataset.data_dir:
        data_dir = cfg.dataset.data_dir

    eval_dataset = tfds.load(
        cfg.dataset.name, split=list(cfg.dataset.split), data_dir=data_dir, as_supervised=True
    )[0]

    eval_dataset = prepare_dataset(
        eval_dataset, mode=cfg.evaluate_tflite.pp_mode, batch_size=1, is_training=False
    )

    # Step3: evaluate

    metric = tf.keras.metrics.get(cfg.evaluate_tflite.metric)
    for data, label in eval_dataset.as_numpy_iterator():
        interpreter.set_tensor(input_details[0]["index"], data)

        interpreter.invoke()

        predict = interpreter.get_tensor(output_details[0]["index"])
        metric.update_state([label], predict)

    print(metric.result())


if __name__ == "__main__":
    evaluate_tflite()
