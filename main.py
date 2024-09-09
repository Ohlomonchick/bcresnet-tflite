# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
from argparse import ArgumentParser
import shutil
from glob import glob
# import onnxruntime as ort

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from utils import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset
import ai_edge_torch
os.environ["PJRT_DEVICE"] = "CPU"

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch._export import capture_pre_autograd_graph

from ai_edge_torch.quantize.pt2e_quantizer import get_symmetric_quantization_config
from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
from ai_edge_torch.quantize.quant_config import QuantConfig

import tensorflow as tf



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
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        parser.add_argument("--download", help="download data", action="store_true")
        parser.add_argument("--onnx_path", default="model.onnx", help="path to save the ONNX model", type=str)
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        # self.device = xm.xla_device()
        self._load_data()
        self._load_model()

    def __call__(self):
        """
        Method that allows the object to be called like a function.

        Trains the model and presents the train/test progress.
        """
        # train hyperparameters
        total_epoch = 1
        warmup_epoch = 5
        init_lr = 1e-1
        lr_lower_limit = 0

        # optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        # train
        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                # lr cos schedule
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

            # valid
            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))

        test_acc = self.Test(self.test_dataset, self.test_loader, augment=False)  # official testset
        print("test acc: %.3f" % (test_acc))
        print("End.")

        self.export_to_onnx()
        self.export_to_tflite()
        self.export_to_tflite_quant_dynamic()

    def Test(self, dataset, loader, augment):
        """
        Tests the model on a given dataset.

        Parameters:
            dataset (Dataset): The dataset to test the model on.
            loader (DataLoader): The data loader to use for batching the data.
            augment (bool): Flag indicating whether to use data augmentation during testing.

        Returns:
            float: The accuracy of the model on the given dataset.
        """
        true_count = 0.0
        num_testdata = float(len(dataset))
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0  # percentage
        return acc

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
        train_dir = "%s/train_12class" % base_dir
        valid_dir = "%s/valid_12class" % base_dir
        noise_dir = "%s/_background_noise_" % base_dir
        self.noise_dir = noise_dir

        transform = transforms.Compose([Padding()])
        self.train_dataset = SpeechCommand(train_dir, self.ver, transform=transform)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = SpeechCommand(valid_dir, self.ver, transform=transform)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=100, num_workers=0)
        self.test_dataset = SpeechCommand(test_dir, self.ver, transform=transform)
        self.test_loader = DataLoader(self.test_dataset, batch_size=100, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        # Define preprocessors
        self.preprocess_train = Preprocess(
            noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(noise_dir, self.device)

    def _load_model(self):
        """
        Private method that loads the model into the object.
        """
        print("model: BC-ResNet-%.1f on data v0.0%d" % (self.tau, self.ver))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)

    def representative_dataset(self):
        for batch_data, labels in self.valid_loader:
            if len(labels) == 100:
                inputs = batch_data.to(self.preprocess_test.device)
                labels = labels.to(self.preprocess_test.device)
                inputs = self.preprocess_test(inputs, labels, augment=True)
                inputs.to(self.device)
                yield [inputs.cpu().numpy()]


    def export_to_tflite(self, quant=True):
        self.device = torch.device("cpu")
        dummy_input = torch.randn(100, 1, 40, 101, device=self.device, dtype=torch.float32)  # Adjust the size
        self.model.to(self.device)
        sample_inputs = (dummy_input,)
        self.preprocess_test = Preprocess(self.noise_dir, self.device)
        if quant:
            tfl_converter_flags = {
                'optimizations': [tf.lite.Optimize.DEFAULT],
                'inference_input_type': tf.uint8,
                'inference_output_type': tf.uint8,
                'representative_dataset': self.representative_dataset
            }
        else:
             tfl_converter_flags = {}

        edge_model = ai_edge_torch.convert(
            self.model.eval(), sample_inputs, _ai_edge_converter_flags=tfl_converter_flags
        )
        edge_model.export("edge.tflite")

    def export_to_tflite_quant_dynamic(self):
        self.device = torch.device("cpu")
        dummy_input = torch.randn(100, 1, 40, 101, device=self.device, dtype=torch.float32)  # Adjust the size
        self.model.to(self.device)
        sample_inputs = (dummy_input,)

        pt2e_quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )

        pt2e_torch_model = capture_pre_autograd_graph(self.model, sample_inputs)
        pt2e_torch_model = prepare_pt2e(pt2e_torch_model, pt2e_quantizer)
        pt2e_torch_model(*sample_inputs)

        pt2e_torch_model = convert_pt2e(pt2e_torch_model, fold_quantize=False)


        edge_model = ai_edge_torch.convert(pt2e_torch_model, sample_inputs, quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer))
        edge_model.export("edge_quant.tflite")


    def export_to_onnx(self):
        self.device = torch.device("cpu")
        self.model.to(self.device)
        dummy_input = torch.randn(100, 1, 40, 101, device=self.device, dtype=torch.float32)  # Adjust the size
        torch.onnx.export(self.model, dummy_input, self.onnx_path, opset_version=15)
        print(f"Model has been converted to ONNX and saved at {self.onnx_path}")


if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()
