import multiprocessing

import hydra
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from ktk.utils import tf_set_memory_growth
from omegaconf import DictConfig
from tensorflow import keras


# TODO modularize preprocessing
def preprocess_dataset(shape, mode="tf", is_training=True):
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


def prepare_dataset(dataset, shape, mode="tf", batch_size=32, is_training=True):
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(
        preprocess_dataset(shape, mode, is_training), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@hydra.main(version_base="1.2", config_path="./config", config_name="evaluate_tflite")
def evaluate_tflite(cfg: DictConfig) -> None:
    tf_set_memory_growth()

    # Step1: Load a tflite model
    cpu_num = multiprocessing.cpu_count()
    if cpu_num > 16:
        cpu_num = 16

    interpreter = tf.lite.Interpreter(
        model_path=cfg.evaluate_tflite.model_path, num_threads=cpu_num
    )
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

    # TODO: handle multiple inputs
    input_shape = input_details[0]["shape"][1:3]
    eval_dataset = prepare_dataset(
        eval_dataset,
        shape=input_shape,
        mode=cfg.evaluate_tflite.pp_mode,
        batch_size=1,
        is_training=False,
    )

    # Step3: evaluate

    num_elements = tf.data.experimental.cardinality(eval_dataset).numpy()
    bar = tqdm.tqdm(total=num_elements)
    metric = tf.keras.metrics.get(cfg.evaluate_tflite.metric)
    for data, label in eval_dataset.as_numpy_iterator():
        interpreter.set_tensor(input_details[0]["index"], data)

        interpreter.invoke()

        predict = interpreter.get_tensor(output_details[0]["index"])
        metric.update_state([label], predict)
        bar.update(1)

    print(metric.result())


if __name__ == "__main__":
    evaluate_tflite()
