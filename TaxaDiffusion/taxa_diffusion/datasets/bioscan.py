import os
import random
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

class BioScanDataset(Dataset):
    def __init__(self, csv_file, base_url_image, fallback_img_path, mapping_file, img_size=512, training=True, threshold=.5):
        self.annotations = pd.read_csv(csv_file)
        if not training:
            self.annotations = self.annotations.head(20)
        self.base_url_image = base_url_image
        self.fallback_img_path = fallback_img_path
        self.mapping_file = mapping_file
        self.resolution = img_size
        self.training = training
        self.threshold = threshold
        self.condition_mappings = self.load_or_create_mappings()
    
    def __len__(self):
        return len(self.annotations)
    
    def load_or_create_mappings(self):
        if os.path.exists(self.mapping_file):
            return self.load_mappings()
        else:
            mappings = self.create_mappings()
            self.save_mappings(mappings)
            return mappings
        
    def create_mappings(self):
        mappings = {
            'class': {}, 
            'order': {}, 'family': {}, 'genus': {}, 
            'specific_epithet': {}
        }
        for _, row in self.annotations.iterrows():
            category = {
                'class': row['class'],
                'order': row['order'],
                'family': row['family'],
                'genus': row['genus'],
                'specific_epithet': row['species'],
            }
            for key, value in category.items():
                if value not in mappings[key]:
                    mappings[key][value] = len(mappings[key])

        # for key in mappings.keys():
        #     mappings[key]['None'] = len(mappings[key])
        
        return mappings

    def save_mappings(self, mappings):
        with open(self.mapping_file, 'w') as f:
            for key, mapping in mappings.items():
                f.write(f"{key}\n")
                for value, index in mapping.items():
                    f.write(f"{value}: {index}\n")
                f.write("\n")

    def load_mappings(self):
        mappings = {
            'class': {}, 
            'order': {}, 'family': {}, 'genus': {}, 
            'specific_epithet': {}
        }
        with open(self.mapping_file, 'r') as f:
            current_key = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line in mappings:
                    current_key = line
                elif current_key:
                    value, index = line.split(': ')
                    mappings[current_key][value] = int(index)
        return mappings
    
    def __getitem__(self, index):
        annotation = self.annotations.iloc[index]
        img_info = {
            'file_name': annotation['image_file'],
            'image_id': annotation['chunk_number'],
            'sampleid': annotation['sampleid']
        }
        
        # Load image or fallback image if not available
        img_path = self.get_image_path(annotation)
        note = "Image available"
        error_loaded_image = False
        
        try:
            image = self.imread(img_path)
        except (FileNotFoundError, IOError):
            image = Image.open(os.path.join(self.base_url_image, self.fallback_img_path)).convert('RGB')
            image = np.array(image)
            note = "Image not available, using fallback image"
            error_loaded_image = True

        # Resize image based on training or evaluation mode
        image = self.resize_image_with_padding(image, self.resolution)
        
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        
        # Get category info
        category = {
            'class': annotation['class'],
            'order': annotation['order'],
            'family': annotation['family'],
            'genus': annotation['genus'],
            'specific_epithet': annotation['species'],
        }

        # Generate conditions and category info
        chance = [0, 0, 0, 0, 0]
        random_number = random.randint(0, 20)

        if random_number == 0:
            chance = [0, 0, 0, 0, 1]
        elif random_number == 1:
            chance = [0, 0, 0, 1, 0]
        elif random_number == 2:
            chance = [0, 0, 1, 0, 0]
        elif random_number == 3:
            chance = [0, 1, 0, 0, 0]
        elif random_number == 4:
            chance = [1, 0, 0, 0, 0]

        conditions_list_name = [
            f"class: {annotation.get('class', 'None')}" if chance[0] == 0 else "class: None",
            f"order: {annotation.get('order', 'None')}" if chance[1] == 0 else "order: None",
            f"family: {annotation.get('family', 'None')}" if chance[2] == 0 else "family: None",
            f"genus: {annotation.get('genus', 'None')}" if chance[3] == 0 else "genus: None",
            f"specific_epithet: {annotation['species']}" if chance[4] == 0 else "specific_epithet: None",
        ]

        conditions, conditions_list, cut_off_index = self.map_condition(category)
        conditions_list = torch.as_tensor(conditions_list)

        # Create formatted name
        name = self.create_name(category, cut_off_index) + "__" + img_info['sampleid']
        if error_loaded_image:
            name = "_1" + "__" + img_info['sampleid']

        name = name.replace('/', '_')
        
        return {
            'target_image': image,
            'label': 1,
            'conditions': conditions,
            'conditions_list': conditions_list,
            'note': note,
            'name': name,
            'prompt': " ".join(conditions_list_name),
            'conditions_list_name': conditions_list_name
        }
    
    def get_image_path(self, row):
        return os.path.join(self.base_url_image, "part" + str(row['chunk_number']), row['image_file'])
    
    def resize_image_with_padding(self, image, resolution, padding_color=(255, 255, 255)):
        """
        Resize an image to a square shape with padding and then to a fixed resolution.

        Parameters:
            image (numpy array): Input image in HWC format.
            resolution (int): Desired resolution (e.g., 512 for 512x512).
            padding_color (tuple): RGB color for the padding (default is black).

        Returns:
            numpy array: Resized image with padding applied.
        """
        H, W, C = image.shape

        # Determine the size of the square image
        size = max(H, W)

        # Create a square canvas with the padding color
        padded_image = np.full((size, size, C), padding_color, dtype=image.dtype)

        # Center the original image on the square canvas
        y_offset = (size - H) // 2
        x_offset = (size - W) // 2
        padded_image[y_offset:y_offset + H, x_offset:x_offset + W] = image

        # Resize the padded image to the desired resolution
        resized_image = cv2.resize(padded_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
        
        return resized_image
    
    def imread(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode == 'PA' or img.mode == 'P':
                img = img.convert('RGBA')
            return np.asarray(img.convert('RGB'))
        except UnidentifiedImageError:
            print(f"Cannot identify image file {image_path}")
        except OSError as e:
            print(f"Error processing file {image_path}: {e}")
        return None

    def map_condition(self, category):
        conditions = {}
        conditions_list = []
        cut_off_index = 0
        for key in self.condition_mappings.keys():
            value = category[key]
            conditions[key] = self.condition_mappings[key].get(value, len(self.condition_mappings[key])-1)
            conditions_list.append(conditions[key])
            cut_off_index += 1
        return conditions, conditions_list, cut_off_index

    def create_name(self, category, cut_off_index):
        name_parts = [str(category[key]) for key in ['class', 'order', 'family', 'genus', 'specific_epithet']]
        name = ""
        for i in range(len(name_parts)):
            if i < cut_off_index:
                name = name + name_parts[i] + "_"
            else:
                name = name + "None_"
        return name


def test_bio_scan_dataset():
    # Define paths and parameters for the test
    csv_file = '/home/karimimonsefi.1/TaxonomyGen/bio_scan_data/data.csv'  # Replace with the path to your CSV file
    base_url_image = '/home/karimimonsefi.1/BIOSCAN/bioscan/images/cropped_full'    # Replace with the base URL where images are stored
    fallback_img_path = 'fallback.jpg'    # Replace with the name of the fallback image
    mapping_file = '/home/karimimonsefi.1/TaxonomyGen/bio_scan_data/mappings.txt' # Replace with the path to save/load the mappings
    resolution = 512                      # Desired image resolution for the dataset

    # Initialize the dataset
    dataset = BioScanDataset(
        csv_file=csv_file,
        base_url_image=base_url_image,
        fallback_img_path=fallback_img_path,
        mapping_file=mapping_file,
        img_size=resolution,
        training=True  # Set to True for training mode, False for validation/testing
    )
    print(len(dataset))
    # Iterate over a few samples to test functionality
    for i in range(30):
        try:
            sample = dataset[i]
            print(f"Sample {i}:")
            print(f"  Image Shape: {sample['target_image'].shape}")
            print(f"  Label: {sample['label']}")
            print(f"  Conditions: {sample['conditions']}")
            print(f"  Conditions List: {sample['conditions_list']}")
            print(f"  Note: {sample['note']}")
            print(f"  Name: {sample['name']}")
            print(f"  Conditions List Name: {sample['conditions_list_name']}")
            print('-' * 50)
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
