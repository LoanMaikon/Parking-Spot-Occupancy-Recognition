from torchvision.transforms import v2
import torch
import yaml
import torch.optim as optim
import shutil
import matplotlib.pyplot as plt
import numpy as np
import os
from time import strftime, localtime

from .simclr_resnet50 import simclr_resnet50
from .custom_dataset import custom_dataset
from .nt_xent import nt_xent
from .lars import LARS

class Model():
    def __init__(
                self,
                operation,
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
        self._set_device()
        self._create_output_dir()
        self._load_model()

        self._load_transform()
        self._load_dataloaders()
        self._load_criterion()
        self._load_optimizer()

        if self.use_scheduler:
            self._load_scheduler()
    
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
    
    def get_mode(self):
        return self.mode

    def get_save_every(self):
        return self.save_every

    def model_infer(self, batch):
        x = torch.cat([batch[0], batch[1]], dim=0).to(self.device)

        return self.model(x)

    def apply_criterion(self, preds, batch):
        z1, z2 = preds.chunk(2, dim=0)

        return self.criterion(z1, z2)

    def get_learning_rate(self):
        if self.optimizer_type == "lars":
            return self.base_optimizer.param_groups[0]['lr']
        else:
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
        
        self.train_transform = v2.Compose([
            v2.RandomResizedCrop(self.transform_resize),
            v2.RandomHorizontalFlip(0.5),
            __get_color_distortion(),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.val_transform = v2.Compose([
            v2.Resize(self.transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_dataloaders(self):
        train_dataset = custom_dataset(
            dataset_folder_path=self.dataset_path,
            data=self.train_data,
            first_n_days=self.first_n_days_train,
            transform=self.train_transform,
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

    def _load_criterion(self):
        self.criterion = nt_xent(self.temperature)

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
        def __lr_lambda(current_epoch_or_step):
            current_epoch_or_step += 1
            if current_epoch_or_step < self.warmup_steps:
                return float(current_epoch_or_step) / float(max(1, self.warmup_steps))

            progress = (current_epoch_or_step - self.warmup_steps) / (self.num_steps - self.warmup_steps)

            return 0.5 * (1. + np.cos(np.pi * progress)) # Cosine decay formula
        
        match self.optimizer_type:
            case "adamw":
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=__lr_lambda)

            case "lars":
                self.scheduler = optim.lr_scheduler.LambdaLR(self.base_optimizer, lr_lambda=__lr_lambda)

            case "sgd":
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=__lr_lambda)

    def _load_model(self):
        if not os.path.exists(self.encoder_weights):
            raise Exception(f"Encoder weights path {self.encoder_weights} does not exist.")

        self.model = simclr_resnet50(use_checkpoint=self.use_checkpoint)
        self.model.load_weights(self.encoder_weights, self.device)

        self.model.fit_projection_head()

        match self.unfrozen_layers:
            case "all":
                for param in self.model.parameters():
                    param.requires_grad = True
            case "classifier_head":
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            case "classifier_head_and_last_block":
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            case "classifier_head_and_last_two_blocks":
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.layer3.parameters():
                    param.requires_grad = True
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            case _:
                raise ValueError(f"Unfrozen layers option {self.unfrozen_layers} not recognized.")

        self.model.to(self.device)

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
        self.first_n_days_train = list(self.config['first_n_days_train'])
        self.batch_size = int(self.config['batch_size'])
        self.num_steps = int(self.config['num_steps'])
        self.lr = float(self.config['lr'])
        self.weight_decay = float(self.config['weight_decay'])
        self.num_workers = int(self.config['num_workers'])
        self.prefetch_factor = int(self.config['prefetch_factor'])
        self.transform_resize = list(self.config['transform_resize'])
        self.temperature = float(self.config['temperature'])
        self.warmup_steps = int(self.config['warmup_steps'])
        self.pin_memory = bool(self.config['pin_memory'])
        self.use_checkpoint = bool(self.config['use_checkpoint'])
        self.optimizer_type = str(self.config['optimizer'])
        self.save_every = int(self.config['save_every'])
        self.use_scheduler = bool(self.config['use_scheduler'])
        self.unfrozen_layers = str(self.config['unfrozen_layers'])
        self.persistent_workers = bool(self.config['persistent_workers'])

        self.dataset_path += '/' if not self.dataset_path.endswith('/') else ''

    def write_on_log(self, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())

        mode = "w" if not os.path.exists(os.path.join(self.output_path, "log.txt")) else "a"

        with open(os.path.join(self.output_path, "log.txt"), mode) as file:
            file.write(time + text + "\n")
