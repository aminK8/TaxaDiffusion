import os
import random
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

class FishDataset(Dataset):
    def __init__(self, csv_file, base_url_image, bbox_json, fallback_img_path, mapping_file, img_size=512, training=True, threshold=.5):
        self.annotations = pd.read_csv(csv_file)
        if not training:
            self.annotations = self.annotations.head(20)
        self.base_url_image = base_url_image
        self.bbox_data = self.load_bbox_data(bbox_json)
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
                'class': row['Class'],
                'order': row['Order'],
                'family': row['Family'],
                'genus': row['Genus'],
                'specific_epithet': row['species'] if not pd.isna(row['species']) else 'None'
            }
            for key, value in category.items():
                if value not in mappings[key]:
                    mappings[key][value] = len(mappings[key])

        for key in mappings.keys():
            mappings[key]['None'] = len(mappings[key])
        
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
    
    def crop_image_with_bbox(self, image, bbox):
        """Crop the image using bounding box coordinates (x, y, w, h)."""
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        return image[y:y + h, x:x + w]
    
    def __getitem__(self, index):
        annotation = self.annotations.iloc[index]
        img_info = {
            'file_name': annotation['image'],
            'image_id': annotation['image'][:-4].replace('https://www.fishbase.se/images/species/', '')
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

        # Apply BBox cropping if available
        if img_info['image_id'] in self.bbox_data:
            bbox = self.bbox_data[img_info['image_id']]
            image = self.crop_image_with_bbox(image, bbox)

        # Resize image based on training or evaluation mode
        if self.training:
            image = self.resize_image_random_cropping(image, self.resolution)
        else:
            image = self.resize_image_fixed_center_cropping(image, self.resolution)
        
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        
        # Get category info
        category = {
            'class': annotation['Class'],
            'order': annotation['Order'],
            'family': annotation['Family'],
            'genus': annotation['Genus'],
            'specific_epithet': annotation['species'] if not pd.isna(annotation['species']) else 'None'
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
            f"class: {annotation.get('Class', 'None')}" if chance[0] == 0 else "class: None",
            f"order: {annotation.get('Order', 'None')}" if chance[1] == 0 else "order: None",
            f"family: {annotation.get('Family', 'None')}" if chance[2] == 0 else "family: None",
            f"genus: {annotation.get('Genus', 'None')}" if chance[3] == 0 else "genus: None",
            f"specific_epithet: {annotation['species'] if not pd.isna(annotation['species']) else 'None'}" if chance[4] == 0 else "specific_epithet: None",
        ]

        conditions, conditions_list, cut_off_index = self.map_condition(category)
        conditions_list = torch.as_tensor(conditions_list)

        # Create formatted name
        name = self.create_name(category, cut_off_index) + "__" + img_info['image_id']
        if error_loaded_image:
            name = "_1" + "__" + img_info['image_id']

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

    def load_bbox_data(self, bbox_json):
        """Load bounding box data from a JSON file."""
        with open(bbox_json, 'r') as f:
            return json.load(f)
    
    def get_image_path(self, row):
        image_name = row['image']
        if 'https://www.fishbase.se/images/species/' in image_name:
            image_name = image_name.replace('https://www.fishbase.se/images/species/', '')
        return os.path.join(self.base_url_image, row['Folder'], image_name)
    
    def resize_image_random_cropping(self, image, resolution):
        H, W, C = image.shape
        if W >= H:
            crop_l = random.randint(0, W - H)
            crop_r = crop_l + H
            crop_t = 0
            crop_b = H
        else:
            crop_t = random.randint(0, H - W)
            crop_b = crop_t + W
            crop_l = 0
            crop_r = W
        
        if random.random() > 1 - self.threshold:
            image = image[crop_t:crop_b, crop_l:crop_r]
        k = float(resolution) / min(H, W)
        img = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

    def resize_image_fixed_center_cropping(self, image, resolution):
        """Resize the image using fixed center cropping, used for validation/testing."""
        H, W, C = image.shape
        if W >= H:
            crop_l = (W - H) // 2
            crop_r = crop_l + H
            crop_t = 0
            crop_b = H
        else:
            crop_t = (H - W) // 2
            crop_b = crop_t + W
            crop_l = 0
            crop_r = W
        
        image = image[crop_t:crop_b, crop_l:crop_r]
        k = float(resolution) / min(H, W)
        img = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img
    
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
