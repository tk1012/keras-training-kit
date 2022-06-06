# A template project for kaggle competitions based on keras

## Run a docker image

Use `vscode` and open the root directory of this project.
Then, you will see a popup to reopen in a dev container.

## Create a virutal env in the dev container

```sh
poetry install
poetry shell
```

## Download a kaggle competition dataset

```sh
mkdir kaggle_data
cd kaggle_data
kaggle competitions download -c <competition name>
```

## Create your custom tfds dataset

```sh
mkdir datasets
cd datasets/
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset.py` to implement your dataset.
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`
```

```python
# in your python code
import tensorflow_datasets as tfds
import datasets.my_dataset

ds = tfds.load("my_dataset")
```


## Train a model

```sh
# modify ktk/apps/config/train.yaml
python -m ktk.apps.train

# or specify your own config
python -m ktk.apps.train --config_name=myconf
```

## Evaluate a model

```sh
# modify ktk/apps/config/evaluate.yaml
python -m ktk.apps.evaluate

# or specify your own config
python -m ktk.apps.evaluate --config_name=myconf
```

## Convert a saved model into a tflite mode and evaluate the converted model

```sh
python -m ktk.apps.tflite_converter
python -m ktk.apps.evaluate_tflite
```

## Imagenet Keras Application

### Save a pre-trained model from tf.keras.applications

```sh
python -m ktk.apps.save_keras.apps
```

### Evaluate

```sh
python -m ktk.apps.evaluate --config-name=evaluate_keras_app
```

### Convert into a TFLite model and evaluate it

```sh
python -m ktk.apps.tflite_converter --config-name=tflite_converter_keras_app
python -m ktk.apps.evaluate_tflite --config-name=evaluate_tflite_keras_app
```
