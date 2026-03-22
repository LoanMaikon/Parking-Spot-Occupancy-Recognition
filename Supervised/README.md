<div align="center">

# IJCNN PARKINNG SPOT OCCUPANCY RECOGNITION - THIRD STAGE (3)

This folder contains the code for the third stage (3) of the training pipeline described in the IJCNN 2026 paper *Toward Parking Spot Occupancy Recognition: A Self-supervised Approach*. This stage involves supervised fine-tuning of the model trained on stage (2) on parking spot occupancy data using the PKLot, CNRPark-EXT, and PLds datasets.

</div>

## 1. Overview

This repository contains:
- `finetune.py`: Main code for fine-tuning of the encoder.
- `src/custom_dataset.py`: Custom dataset class for loading parking spot occupancy data.
- `src/lars.py`: LARS optimizer implementation.
- `src/Model`: Contains the heart of the propposed method. Here, every object is created.
- `src/nt_xent.py`: NT-Xent loss implementation.
- `src/simclr_resnet50.py`: ResNet-50 backbone for SimCLR.
- `src/metrics.py`: Metrics for evaluation.
- `configs/`: Configuration files for training.

## 2. Training

To fine-tune the model on parking spot occupancy data in a supervised manner, follow these steps

### 2.1 Adapt Configuration Files

In `configs/`, you can find the configuration files for the proposed approaches, for the Self-supervised Baseline and the Supervised Baseline. For example, the `configs/proposed_approaches/specialized_camera1.yaml` is the configuration file used to train the Specialized Model for camera1. Each config file contain the hyperparameter settings used in the paper, and follow the format below.

<pre>
mode: supervised # TRAINING MODE: 'supervised' FOR SUPERVISED FINE-TUNING
dataset_path: ../../dataset # PATH TO THE FOLDER CONTAINING THE DATASETS AS SPECIFIED IN `tools/generate_dataset.py`
train_data: ['pklot', 'plds'] # DATASETS USED FOR TRAINING
test_data: [['camera1']] # DATASETS USED FOR TESTING. NOTE THAT EACH ELEMENT OF THE LIST IS A LIST ITSELF, ALLOWING MULTIPLE TEST SETS (E,G., [['camera1'], ['camera2'], ['camera3']])
first_n_days_train: ['all', 'all'] # NUMBER OF DAYS USED FOR TRAINING FOR EACH DATASET (e.g., 'all' OR AN INTEGER VALUE)
first_n_days_test: [['>7']] # NUMBER OF DAYS USED FOR TESTING. MUST HAVE THE SAME STRUCTURE AS test_data (e.g., [['>7'], ['all'], [5]])
n_train_labels: ['all', 'all'] # NUMBER OF LABELED SAMPLES USED FOR TRAINING FOR EACH DATASET (e.g., 'all' OR AN INTEGER VALUE)
batch_size: 256 # BATCH SIZE FOR TRAINING
num_steps: 30000 # NUMBER OF TRAINING STEPS
lr: 0.031622776 # LEARNING RATE
weight_decay: 0.00001 # WEIGHT DECAY
num_workers: 32 # NUMBER OF WORKERS FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
prefetch_factor: 4 # PREFETCH FACTOR FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
transform_resize: [224, 224] # IMAGE RESIZE DIMENSIONS
warmup_steps: 0 # WARMUP STEPS FOR LEARNING RATE SCHEDULER
pin_memory: False # PIN MEMORY FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
use_checkpoint: False # USE CHECKPOINTING TO REDUCE VRAM MEMORY USAGE, ADAPT ACCORDING TO YOUR HARDWARE
optimizer: sgd # OPTIMIZER TYPE (e.g., 'lars', 'adamw' or 'sgd')
save_every: 0 # SAVE CHECKPOINT EVERY N STEPS. SET TO 0 TO DISABLE CHECKPOINT SAVING
use_scheduler: False # COSINE LR SCHEDULER USAGE
inference_batch_size: 256 # BATCH SIZE FOR INFERENCE
data_augmentations: [] # DATA AUGMENTATIONS TO APPLY DURING TRAINING (E.G., ['random_resized_crop', 'horizontal_flip', 'color_jitter', 'grayscale', 'gaussian_blur'])
classifier_head: linear_classifier # TYPE OF CLASSIFIER HEAD (E.G., 'linear_classifier' OR 'mlp_classifier')
unfrozen_layers: all # WHICH LAYERS TO UNFREEZE DURING FINE-TUNING (e.g., 'all', 'classifier_head', 'classifier_head_and_last_block', or 'classifier_head_and_last_two_blocks')
persistent_workers: False # WHETHER TO USE PERSISTENT WORKERS FOR DATA LOADING, ADAPT ACCORDING TO YOUR HARDWARE
</pre>

### 2.2 Run Fine-tuning

To start the fine-tuning process, run the following command:

```bash
python finetune.py --config_file <path_to_config_file> --output_path <path_to_save> --gpu <gpu_index> --encoder_weights <path_to_pretrained_weights>
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

If you want to fine-tune for evaluating one of the baselines, change --encoder_weights <path_to_pretrained_weights>. Put `imagenet` for the Supervised Baseline, and put the path to the SimCLR pretrained weights for the Self-supervised Baseline. If you want to evaluate a Strong General Model or Specialized Model, just put the path to the corresponding to the weights trained in the (2) stage.

### 2.3 Evaluation

To evaluate the fine-tuned model, you can use the `test.py` as follows:

```bash
python test.py --train_dir <path_to_finetuned_model_directory> --gpu <gpu_index>
```

The `path_to_finetuned_model_directory` should point to the directory where the fine-tuned model weights are saved (i.e., the `<path_to_save>` used in the fine-tuning step). This will create a folder such as:

```
test
└── test_camera1_>7
    ├── config.yaml
    ├── inference.json
    ├── log.txt
    └── metrics.json
```

The `inference.json` file contains the model predictions for each sample in the test set, while the `metrics.json` file contains the evaluation metrics (accuracy, macro f1 and loss) for the test set.
