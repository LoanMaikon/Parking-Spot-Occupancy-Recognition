import torch.nn as nn
import argparse
import torch
import os

from src.Model import Model

def main():
    args = get_args()

    model = Model(
                 operation="test",
                 config_path=f"{args.train_dir}/config.yaml",
                 gpu_index=args.gpu,
                 output_path=f"{args.train_dir}/test",
                 encoder_weights=f"{args.train_dir}/model.pth",
                 )
    
    for _ in range(model.get_test_index(), model.get_max_test_index() + 1):
        inference(model)
        model.generate_metrics()
        model.step_test_index()

def inference(model):
    model.write_on_log("Starting testing...")

    model.model_to_eval()

    image_paths = []
    targets = []
    predictions = []
    loss = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in model.get_test_dataloader():
            with torch.amp.autocast(device_type='cuda'):
                outputs = model.model_infer(batch)

            preds = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy().tolist() # Probability of class 1 - occupied

            predictions.extend(preds)
            targets.extend(batch[1].cpu().detach().numpy().tolist())
            image_paths.extend(batch[2])

            loss += criterion(outputs, batch[1].to(model.device)).item() * len(batch[1])
            total_samples += len(batch[1])

    loss /= total_samples

    model.save_test_results(image_paths, targets, predictions, loss, total_samples)

    model.write_on_log("Testing completed.")

def handle_args(args):
    args.train_dir += '/' if not args.train_dir.endswith('/') else ''

    if not os.path.exists(args.train_dir):
        raise ValueError(f"Training directory {args.train_dir} does not exist.")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory.')
    parser.add_argument('--gpu', type=int, required=True, help='GPU index to use.')

    args = parser.parse_args()
    handle_args(args)
    return args

if __name__ == "__main__":
    main()
