<div align="center">

# IJCNN PARKINNG SPOT OCCUPANCY RECOGNITION - SECOND STAGE (2)

This folder contains the code for the second stage (2) of the training pipeline described in the IJCNN 2026 paper *Toward Parking Spot Occupancy Recognition: A Self-supervised Approach*. This stage involves self-supervised fine-tuning of a SimCLR model on parking spot occupancy data using the PKLot, CNRPark-EXT, and PLds datasets.

</div>

## 1. Overview

This repository contains:
- `finetune.py`: Main code for fine-tuning of the encoder.
- `src/custom_dataset.py`: Custom dataset class for loading parking spot occupancy data.
- `src/lars.py`: LARS optimizer implementation.
- `src/Model`: Contains the heart of the propposed method. Here, every object is created.
- `src/nt_xent.py`: NT-Xent loss implementation.
- `src/simclr_resnet50.py`: ResNet-50 backbone for SimCLR.
- `configs/`: Configuration files for training.

## 2. Training

To fine-tune the SimCLR model on parking spot occupancy data, follow these steps

### 2.1 Adapt Configuration Files

For example, the `configs/general_finetune_encoder_cnr_plds.yaml` is the configuration file used to fine-tune the Strong General model on the CNRPark-EXT and PLds datasets for evaluation on PKLot. You can modify the parameters in this file according to your requirements as follows:

<pre>
mode: simclr # TRAINING MODE: 'simclr' FOR SIMCLR FINE-TUNING
dataset_path: ../../dataset # PATH TO THE FOLDER CONTAINING THE DATASETS AS SPECIFIED IN `tools/generate_dataset.py`
train_data: ['cnr', 'plds'] # DATASETS USED FOR TRAINING
first_n_days_train: ['all', 'all'] # NUMBER OF DAYS USED FOR TRAINING FOR EACH DATASET (e.g., 'all' OR AN INTEGER VALUE)
batch_size: 512 # BATCH SIZE FOR TRAINING
num_steps: 80000 # NUMBER OF TRAINING STEPS
lr: 0.3 # LEARNING RATE
weight_decay: 0.000001 # WEIGHT DECAY
num_workers: 32 # NUMBER OF WORKERS FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
prefetch_factor: 4 # PREFETCH FACTOR FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
transform_resize: [224, 224] # IMAGE RESIZE DIMENSIONS
temperature: 0.5 # TEMPERATURE PARAMETER FOR NT-XENT LOSS
warmup_steps: 0 # WARMUP STEPS FOR LEARNING RATE SCHEDULER
pin_memory: False # PIN MEMORY FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
use_checkpoint: False # USE CHECKPOINTING TO REDUCE VRAM MEMORY USAGE, ADAPT ACCORDING TO YOUR HARDWARE
optimizer: lars # OPTIMIZER TYPE (e.g., 'lars', 'adamw' or 'sgd')
save_every: 20000 # SAVE CHECKPOINT EVERY N STEPS. SET TO 0 TO DISABLE CHECKPOINT SAVING
use_scheduler: False # COSINE LR SCHEDULER USAGE
unfrozen_layers: all # WHICH LAYERS TO UNFREEZE DURING FINE-TUNING (e.g., 'all', 'classifier_head', 'classifier_head_and_last_block', or 'classifier_head_and_last_two_blocks')
persistent_workers: False # PERSISTENT WORKERS FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
</pre>

### 2.2 Run Fine-tuning

To start the fine-tuning process, run the following command:

```bash
python finetune.py --config_file <path_to_config_file> --output_path <path_to_save> --gpu <gpu_index> --encoder_weights <path_to_pretrained_simclr_weights>
```

This will create an output folder at `<path_to_save>` containing the fine-tuned model weights, training logs, and loss/learning rate plots.

```
config.yaml
figs
├── learning_rate.png
└── train_loss.png
log.txt
model.pth
```

The same applies to the Specialized Model in the Two-stage Deployment scheme. Just use the appropriate configuration file. In addition, if `use_checkpoint` is False, this should run in a GPU with 48 GB of VRAM. If you want to reduce VRAM usage to run in a 32 VRAM GPU, set `use_checkpoint` to True, but this will increase training time.
