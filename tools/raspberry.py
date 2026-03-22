import torchvision.transforms as transforms
import torchvision.models as models
import torch
import time
import json
import cv2
import sys
import os

BATCH_SIZE = 100
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, transform, spots):
        self.images = spots
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image

def main(metapklot, plds, output_path):
    execute(plds, metapklot + "annotations/original/spots/PLds/", "PLds", output_path)
    execute(metapklot, metapklot + "annotations/original/spots/CNRPark-EXT/", "CNRPark-EXT", output_path)
    execute(metapklot, metapklot + "annotations/original/spots/PKLot/", "PKLot", output_path)

def execute(dataset_path, jsons_dir, name, output_path):
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    crop_time = 0
    inference_time = 0
    n_spots = 0
    n_images = 0

    dicts = []

    jsons_path = [f"{jsons_dir}/{f}" for f in os.listdir(jsons_dir) if f.endswith(".json")]
    if len(jsons_path) == 0:
        tar_files = [f"{jsons_dir}/{f}" for f in os.listdir(jsons_dir) if f.endswith(".tar.xz")]
        uncompress_jsons(tar_files)
    
    log_count = 0

    for json_file in jsons_path:
        subset_crop_time = 0
        subset_inference_time = 0
        subset_n_spots = 0
        subset_n_images = 0

        spots_json = json.load(open(json_file, "r"))

        put_annotations_inside_images(spots_json)

        for image in spots_json['images']:
            spots = []

            log_count += 1
            if log_count % 1000 == 0:
                print(f"Processed {log_count} images of dataset {name}")
            
            ##### CROP
            start_crop = time.time()

            image_path = f"{dataset_path}/{image['file_name']}"
            img = cv2.imread(image_path)

            for annotation in image['annotations']:
                best_rotated_rect = annotation['best_rotated_rect']

                center = best_rotated_rect[0]
                size = best_rotated_rect[1]
                angle = best_rotated_rect[2]

                M = cv2.getRotationMatrix2D(center, angle, 1)
                image_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                image_cropped = cv2.getRectSubPix(image_rotated, [int(u) for u in size], center)

                if angle >= 45:
                    image_cropped = cv2.rotate(image_cropped, cv2.ROTATE_90_CLOCKWISE)
                
                spots.append(image_cropped)
            
            end_crop = time.time()
            crop_time += end_crop - start_crop
            subset_crop_time += end_crop - start_crop

            ##### INFERENCE
            start_inference = time.time()

            dataset = custom_dataset(transform, spots)
            batch_size = BATCH_SIZE
            if BATCH_SIZE == "all":
                batch_size = len(spots)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

            model.eval()
            with torch.no_grad():
                for inputs in dataloader:
                    inputs = inputs.to(DEVICE)
                    _ = model(inputs)
                    
            end_inference = time.time()
            inference_time += end_inference - start_inference
            subset_inference_time += end_inference - start_inference
            
            ###############

            n_spots += len(spots)
            subset_n_spots += len(spots)
            n_images += 1
            subset_n_images += 1

        subset_dict = {
            "name": json_file,
            "crop_time": subset_crop_time,
            "inference_time": subset_inference_time,
            "n_spots": subset_n_spots,
            "n_images": subset_n_images,
            "total_time": subset_crop_time + subset_inference_time,
            "time_per_spot": (subset_crop_time + subset_inference_time) / subset_n_spots,
            "time_per_image": (subset_crop_time + subset_inference_time) / subset_n_images,
        }
        dicts.append(subset_dict)
        
    dict = {
        "name": name,
        "crop_time": crop_time,
        "inference_time": inference_time,
        "n_spots": n_spots,
        "n_images": n_images,
        "total_time": crop_time + inference_time,
        "time_per_spot": (crop_time + inference_time) / n_spots,
        "time_per_image": (crop_time + inference_time) / n_images,
    }
    dicts.append(dict)

    os.makedirs(f"{output_path}/results_raspberry", exist_ok=True)
    with open(f"{output_path}/results_raspberry/{name}_results.json", "w") as f:
        json.dump(dicts, f, indent=4)

def put_annotations_inside_images(spots_json):
    images_idx = {image['id']: i for i, image in enumerate(spots_json['images'])}

    for annotation in spots_json['annotations']:

        image_idx = images_idx[annotation['image_id']]
        if spots_json['images'][image_idx].get('annotations') is None:
            spots_json['images'][image_idx]['annotations'] = [annotation]
        else:
            spots_json['images'][image_idx]['annotations'].append(annotation)

def uncompress_jsons(tar_files):
    for tar_file in tar_files:
        os.system(f"tar -xf {tar_file} -C {os.path.dirname(tar_file)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 raspberry.py <metapklot> <plds> <output_path>")
        sys.exit(1)
    
    metapklot = sys.argv[1]
    plds = sys.argv[2]
    output_path = sys.argv[3]

    metapklot += "/" if not metapklot.endswith("/") else ""
    plds += "/" if not plds.endswith("/") else ""

    main(metapklot, plds, output_path)
