import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.callbacks import LearningRateScheduler
from tqdm import tqdm
from torchvision import transforms
import torch

from bcresnet import BCResNets, SubSpectralNorm, ConvBNReLU  # Adjust according to your file structure
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset  # These need to be converted similarly


def preprocess_audio(sample, label, preprocess, labels, is_train):
    sample = sample.clone().detach()
    sample = preprocess(sample.unsqueeze(0), labels, augment=is_train, is_train=is_train)
    return sample.squeeze(axis=0).numpy(), label


def create_tf_dataset(dataset, preprocess, batch_size=1, shuffle_buffer_size=1000, is_train=True):
    def generator():
        for sample, label in dataset:
            yield preprocess_audio(sample, label, preprocess, dataset.labels, is_train)

    output_signature = (
        tf.TensorSpec(shape=(1, 40, 101), dtype=tf.float32),  # Adjust the shape according to preprocessing
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )

    tf_dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if is_train:
        tf_dataset = tf_dataset.shufflex(buffer_size=shuffle_buffer_size)
    tf_dataset = tf_dataset.batch(batch_size)
    return tf_dataset


class Trainer:
    def __init__(self):
        """
        Constructor for the Trainer class.

        Initializes the trainer object with default values for the hyperparameters and data loaders.
        """
        parser = ArgumentParser()
        parser.add_argument(
            "--ver", default=1, help="google speech command set version 1 or 2", type=int
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")

        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = f'/GPU:{self.gpu}' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        self.torch_device = torch.device('gpu') if self.gpu == "1" else torch.device('cpu')
        self._load_data()
        self._load_model()

    def __call__(self):
        correct_predictions = 0
        total_samples = 0

        # Convert the dataset to an iterator
        test_dataset_iterator = iter(self.test_loader)
        self.quant = 'quantization' in self.input_details[0]
        if self.quant:
            input_scale, input_zero_point = self.input_details[0]['quantization']

        for sample in test_dataset_iterator:
            # Assuming the sample is a tuple (input_data, label)
            input_data, labels = sample
            input_data = input_data.numpy()
            labels = labels.numpy()

            # Ensure input_data is the correct shape and type
            # input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension if needed

            if self.quant:
                input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
            else:
                input_data = input_data.astype(np.float32)

            # Set the tensor to point to the input data to be inferred
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            except Exception as e: # if batch is not full
                correct_predictions += len(input_data) * (correct_predictions / total_samples)
                print(self.input_details)
                print(input_data.dtype)
                print(e)
                continue


            # Run the inference
            self.interpreter.invoke()

            # Get the results
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Process the output
            predictions = np.argmax(output_data, axis=1)

            # Calculate the number of correct predictions
            correct_predictions += np.sum(predictions == labels)
            total_samples += len(labels)

        # Calculate accuracy
        accuracy = correct_predictions / total_samples
        print(f"Test Accuracy: {accuracy:.4f}")

    def _load_data(self):
        """
        Private method that loads data into the object.

        Downloads and splits the data if necessary.
        """
        print("Check google speech commands dataset v1 or v2 ...")
        if not os.path.isdir("./data"):
            os.mkdir("./data")
        base_dir = "./data/speech_commands_v0.01"
        url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
        url_test = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
        if self.ver == 2:
            base_dir = base_dir.replace("v0.01", "v0.02")
            url = url.replace("v0.01", "v0.02")
            url_test = url_test.replace("v0.01", "v0.02")
        test_dir = base_dir.replace("commands", "commands_test_set")
        if self.download:
            old_dirs = glob(base_dir.replace("commands_", "commands_*"))
            for old_dir in old_dirs:
                shutil.rmtree(old_dir)
            os.mkdir(test_dir)
            DownloadDataset(test_dir, url_test)
            os.mkdir(base_dir)
            DownloadDataset(base_dir, url)
            SplitDataset(base_dir)
            print("Done...")

        # Define data loaders
        train_dir = f"{base_dir}/train_12class"
        valid_dir = f"{base_dir}/valid_12class"
        noise_dir = f"{base_dir}/_background_noise_"

        # Load datasets using TensorFlow data pipelines
        transform = transforms.Compose([Padding()])
        train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
        valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
        test_dataset = SpeechCommand(test_dir, self.ver, transform=transform)

        self.preprocess_test = Preprocess(noise_dir, self.torch_device)

        self.test_loader = create_tf_dataset(test_dataset, self.preprocess_test, batch_size=100, is_train=False)

        print(f"Check num of data train/valid/test {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path="edge_quant.tflite")
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()