from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import shutil
import matplotlib.pyplot as plt
import numpy as np
import os
from time import strftime, localtime
import json

from .resnet50 import resnet50
from .simclr_resnet50 import simclr_resnet50
from .custom_dataset import custom_dataset
from .lars import LARS
from .metrics import get_accuracy, get_macro_f1

NUM_CLASSES = 2 # PKLot has 2 classes: empty and occupied

class Model():
    def __init__(
                self,
                operation, # Using finetune.py or test.py
                config_path,
                gpu_index,
                output_path,
                encoder_weights,
                ):
        
        self.operation = operation
        self.config_path = config_path
        self.gpu_index = gpu_index
        self.output_path = output_path
        self.encoder_weights = encoder_weights

        self._load_config()
        if self.operation == "test":
            self.test_index = 0
            self.max_test_index = len(self.test_data) - 1

            self.output_path_list = []
            for testd_idx, d in enumerate(self.test_data):
                d_str = f"test_"
                for d_index, di in enumerate(d):
                    d_str += f"{di}_{self.first_n_days_test[testd_idx][d_index]}_"
                d_str = d_str[:-1] + "/"
                self.output_path_list.append(f"{self.output_path}/{d_str}")

            self.output_path = self.output_path_list[self.test_index]

        self._set_device()
        self._create_output_dir()
        self._load_model()

        self._load_transform()
        self._load_dataloaders()
        self._load_criterion()
        self._load_optimizer()

        if self.use_scheduler:
            self._load_scheduler()
    
    def step_test_index(self):
        if self.test_index < self.max_test_index:
            self.test_index += 1
            self.output_path = self.output_path_list[self.test_index]
            self.test_dataloader = self.test_dataloaders[self.test_index]
            os.makedirs(self.output_path, exist_ok=True)

            return True
        return False

    def get_test_index(self):
        return self.test_index
    
    def get_max_test_index(self):
        return self.max_test_index
    
    def get_optimizer(self):
        return self.optimizer

    def get_num_steps(self):
        return self.num_steps
    
    def model_to_train(self):
        self.model.train()

    def model_to_eval(self):
        self.model.eval()

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_val_dataloader(self):
        return self.val_dataloader
    
    def get_test_dataloader(self):
        return self.test_dataloader

    def get_save_every(self):
        return self.save_every

    def model_infer(self, batch):
        x = batch[0].to(self.device)

        return self.model(x)

    def apply_criterion(self, preds, batch):
        labels = batch[1].to(self.device)

        return self.criterion(preds, labels)
                
    def get_predictions_and_targets(self, preds, batch):
        return torch.argmax(preds, dim=1), batch[1].to(self.device)
    
    def get_learning_rate(self):
        if self.optimizer_type == "lars":
            return self.base_optimizer.param_groups[0]['lr']
        return self.optimizer.param_groups[0]['lr']

    def step_scheduler(self):
        if self.use_scheduler:
            self.scheduler.step()
    
    def save_model(self, name=None):
        if name is None:
            name = 'model'

        torch.save(self.model.state_dict(), os.path.join(self.output_path, f"{name}.pth"))

    def plot_fig(self, x, x_name, y, y_name, fig_name):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title(fig_name)

        os.makedirs(self.output_path + "figs", exist_ok=True)

        plt.savefig(os.path.join(self.output_path + "figs", f"{fig_name}.png"))
        plt.close()
    
    def plot_fig_train_val(self, x, x_name, y1, y1_name, y2, y2_name, fig_name):
        plt.figure()
        plt.plot(x, y1, label=y1_name)
        plt.plot(x, y2, label=y2_name)
        plt.xlabel(x_name)
        plt.ylabel("Value")
        plt.title(fig_name)
        plt.legend()

        os.makedirs(self.output_path + "figs", exist_ok=True)

        plt.savefig(os.path.join(self.output_path + "figs", f"{fig_name}.png"))
        plt.close()

    def _load_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])
        
        def __get_augmentations(aug_list):
            aug_transforms = []
            for aug in aug_list:
                match aug:
                    case 'color_distortion':
                        aug_transforms.append(__get_color_distortion(strength=1.0))
                    case 'gaussian_blur':
                        aug_transforms.append(v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5))
                    case 'horizontal_flip':
                        aug_transforms.append(v2.RandomHorizontalFlip(p=0.5))
                    case 'vertical_flip':
                        aug_transforms.append(v2.RandomVerticalFlip(p=0.5))
                    case 'rotation':
                        aug_transforms.append(v2.RandomApply([v2.RandomRotation(degrees=(0, 360))], p=0.5))
                    case 'solarization':
                        aug_transforms.append(v2.RandomSolarize(threshold=0.5, p=0.5))
                    case 'auto_contrast':
                        aug_transforms.append(v2.RandomAutocontrast(p=0.5))
                    case 'random_crop':
                        aug_transforms.append(v2.RandomApply([v2.RandomResizedCrop(size=self.transform_resize)], p=0.5))
                    case _:
                        raise ValueError(f"Data augmentation {aug} not recognized.")
                    
            return aug_transforms


        self.train_transform = v2.Compose([    
            *__get_augmentations(self.data_augmentations),
            
            v2.Resize(self.transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = v2.Compose([
            v2.Resize(self.transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_dataloaders(self):
        if self.operation == "finetune":
            train_dataset = custom_dataset(
                dataset_folder_path=self.dataset_path,
                transform=self.train_transform,
                data=self.train_data,
                first_n_days=self.first_n_days_train,
                n_train_labels=self.n_train_labels,
            )

            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
        
        elif self.operation == "test":
            self.test_dataloaders = []

            for test_index in range(len(self.test_data)):
                self.test_dataset = custom_dataset(
                    dataset_folder_path=self.dataset_path,
                    transform=self.test_transform,
                    data=self.test_data[test_index],
                    first_n_days=self.first_n_days_test[test_index],
                )
            
                self.test_dataloader = torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=self.inference_batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    pin_memory=self.pin_memory,
                    persistent_workers=self.persistent_workers,
                )
                self.test_dataloaders.append(self.test_dataloader)

            self.test_dataloader = self.test_dataloaders[0]

    def _load_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _load_optimizer(self):
        match self.optimizer_type:
            case "adamw":
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            
            case "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                    nesterov=True,
                )

            case "lars":
                self.base_optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.lr,
                    momentum=0.9,
                    weight_decay=self.weight_decay,
                )

                self.optimizer = LARS(self.base_optimizer)

    def _load_scheduler(self):
        def __lr_lambda(current_step):
            current_step += 1
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            progress = (current_step - self.warmup_steps) / (self.num_steps - self.warmup_steps)

            return 0.5 * (1. + np.cos(np.pi * progress)) # Cosine decay formula
        
        match self.optimizer_type:
            case "adamw":
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=__lr_lambda)

            case "lars":
                self.scheduler = optim.lr_scheduler.LambdaLR(self.base_optimizer, lr_lambda=__lr_lambda)

            case "sgd":
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=__lr_lambda)

    def _load_model(self):
        if self.encoder_weights == "imagenet":
            self.model = resnet50(mode="supervised", use_checkpoint=self.use_checkpoint, pretrained=True)

        elif self.encoder_weights == "none":
            self.model = resnet50(mode="supervised", use_checkpoint=self.use_checkpoint, pretrained=False)

        elif os.path.exists(self.encoder_weights):
            try:
                self.model = simclr_resnet50(use_checkpoint=self.use_checkpoint)
                self.model.load_weights(self.encoder_weights, self.device)
            except:
                try:
                    self.model = resnet50(mode="supervised", use_checkpoint=self.use_checkpoint)
                    self.model.load_weights(self.encoder_weights, self.device)
                except:
                    raise ValueError(f"Could not load weights from {self.encoder_weights}.")
        
        else:
            raise ValueError(f"Encoder weights option {self.encoder_weights} not recognized.")
        
        if self.operation == "finetune":
            self.model.fit_classifier_head(NUM_CLASSES)

        if self.operation == "finetune":
            match self.unfrozen_layers:
                case "all":
                    for p in self.model.parameters():
                        p.requires_grad = True

                case "classifier_head":
                    for p in self.model.parameters():
                        p.requires_grad = False

                    for p in self.model.fc.parameters():
                        p.requires_grad = True

                case "classifier_head_and_last_block":
                    for p in self.model.parameters():
                        p.requires_grad = False

                    for p in self.model.layer4.parameters():
                        p.requires_grad = True

                    for p in self.model.fc.parameters():
                        p.requires_grad = True

                case "classifier_head_and_last_two_blocks":
                    for p in self.model.parameters():
                        p.requires_grad = False

                    for p in self.model.layer3.parameters():
                        p.requires_grad = True

                    for p in self.model.layer4.parameters():
                        p.requires_grad = True

                    for p in self.model.fc.parameters():
                        p.requires_grad = True

                case _:
                    raise ValueError(f"Unfrozen layers option {self.unfrozen_layers} not recognized.")
        
        self.model.to(self.device)

    def save_test_results(self, image_paths, targets, predictions, loss, total_samples):
        results = {
            "image_paths": image_paths,
            "targets": targets,
            "predictions": predictions,
            "loss": loss,
            "total_samples": total_samples,
        }

        with open(os.path.join(self.output_path, "inference.json"), "w") as f:
            json.dump(results, f)

    def generate_metrics(self):
        json_file = json.load(open(f"{self.output_path}inference.json", "r"))
        
        predictions = np.array(json_file["predictions"])
        targets = np.array(json_file["targets"])
        image_paths = np.array(json_file["image_paths"])

        accuracy = get_accuracy(predictions, targets, image_paths)
        macro_f1 = get_macro_f1(predictions, targets, image_paths)

        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "loss": json_file["loss"],
            "total_samples": json_file["total_samples"],
        }

        with open(os.path.join(self.output_path, "metrics.json"), "w") as f:
            json.dump(metrics, f)

    def _create_output_dir(self):
        if os.path.exists(self.output_path):
            raise Exception(f"Output path {self.output_path} already exists. Please choose a different path.")
        
        os.makedirs(self.output_path)
        shutil.copy(self.config_path, os.path.join(self.output_path, 'config.yaml'))

    def _set_device(self):
        self.device = torch.device(f'cuda:{self.gpu_index}' if torch.cuda.is_available() else 'cpu')

    def _load_config(self):
        self.config = yaml.safe_load(open(self.config_path, 'r'))

        self.mode = str(self.config['mode'])
        self.dataset_path = str(self.config['dataset_path'])
        self.train_data = list(self.config['train_data'])
        self.test_data = list(self.config['test_data'])
        self.first_n_days_train = list(self.config['first_n_days_train'])
        self.first_n_days_test = list(self.config['first_n_days_test'])
        self.batch_size = int(self.config['batch_size'])
        self.num_steps = int(self.config['num_steps'])
        self.lr = float(self.config['lr'])
        self.weight_decay = float(self.config['weight_decay'])
        self.num_workers = int(self.config['num_workers'])
        self.prefetch_factor = int(self.config['prefetch_factor'])
        self.transform_resize = list(self.config['transform_resize'])
        self.warmup_steps = int(self.config['warmup_steps'])
        self.pin_memory = bool(self.config['pin_memory'])
        self.use_checkpoint = bool(self.config['use_checkpoint'])
        self.optimizer_type = str(self.config['optimizer'])
        self.save_every = int(self.config['save_every'])
        self.use_scheduler = bool(self.config['use_scheduler'])
        self.inference_batch_size = int(self.config['inference_batch_size'])
        self.data_augmentations = list(self.config['data_augmentations'])
        self.classifier_head = str(self.config['classifier_head'])
        self.unfrozen_layers = str(self.config['unfrozen_layers'])
        self.persistent_workers = bool(self.config['persistent_workers'])
        self.n_train_labels = list(self.config['n_train_labels'])

        self.dataset_path += '/' if not self.dataset_path.endswith('/') else ''

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())

        mode = "w" if not os.path.exists(os.path.join(self.output_path, "log.txt")) else "a"

        with open(os.path.join(self.output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")
