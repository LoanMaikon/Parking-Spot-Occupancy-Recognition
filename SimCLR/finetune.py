from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import argparse
import random
import numpy as np
import os

from src.Model import Model

def main():
    args = get_args()

    model = Model(
                 operation="finetune",
                 config_path=args.config,
                 gpu_index=args.gpu,
                 output_path=args.output_path,
                 encoder_weights=args.encoder_weights,
                 )
    
    train(model)
    
def train(model):
    model.write_on_log("Starting training...")

    scaler = torch.amp.GradScaler()

    optimizer = model.get_optimizer()

    train_losses = []
    lrs = []

    actual_step = 1
    while (1):
        for batch in model.get_train_dataloader():
            if actual_step > model.get_num_steps():
                model.write_on_log("Training finished.")
                return
            
            lrs.append(model.get_learning_rate())

            if actual_step % 1000 == 0  or actual_step == model.get_num_steps():
                model.write_on_log(f"Step {actual_step}/{model.get_num_steps()}")

            optimizer.zero_grad()
            model.model_to_train()

            with torch.amp.autocast('cuda'):
                outputs = model.model_infer(batch)

                loss = model.apply_criterion(outputs, batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.step_scheduler()

            loss_value = loss.item()
            train_losses.append(loss_value)

            if actual_step % 1000 == 0  or actual_step == model.get_num_steps():
                model.write_on_log(f"Training loss: {loss_value}")

            if model.get_save_every() > 0:
                if actual_step % model.get_save_every() == 0:
                    model.write_on_log(f"Saving model...")
                    model.save_model(f"model_step_{actual_step}")
            
            if actual_step == model.get_num_steps():
                model.write_on_log(f"Saving final model...")
                model.save_model()

            actual_step += 1
            if actual_step % 1000 == 0  or actual_step == model.get_num_steps():
                model.write_on_log(f"")
                model.plot_fig(range(len(train_losses)), "Training Steps", train_losses, "Training Loss", "train_loss")
                model.plot_fig(range(len(lrs)), "Training Steps", lrs, "Learning Rate", "learning_rate")

def handle_args(args):
    args.output_path += "/" if not args.output_path.endswith("/") else ""

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    if not os.path.exists(args.encoder_weights) and args.encoder_weights not in ["imagenet", "none", "pretrained"]:
        raise FileNotFoundError(f"Encoder weight {args.encoder_weights} does not exist or is invalid.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--gpu', type=int, required=True, help='GPU index to use.')
    parser.add_argument('--encoder_weights', type=str, required=True, help='Encoder weight')

    args = parser.parse_args()
    handle_args(args)
    return args

if __name__ == "__main__":
    main()
