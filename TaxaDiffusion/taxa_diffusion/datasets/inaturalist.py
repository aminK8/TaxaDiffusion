import json
import os
import random
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class INaturalistDataset(Dataset):
    def __init__(self, json_path, img_size=512, fallback_img_path='path/to/fallback_image.jpg', 
                 condition_prob=0.1, mapping_file='condition_mappings.txt', training=True,
                 base_image_url=''):
        
        # Load the JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.img_info = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        self.categories = {cat['id']: cat for cat in data['categories']}
        self.img_size = img_size
        self.condition_prob = condition_prob
        self.mapping_file = mapping_file
        self.fallback_img_path = fallback_img_path
        self.training = training
        self.base_image_url = base_image_url
        
        # Load or create mappings for conditions
        self.condition_mappings = self.load_or_create_mappings()

    def load_or_create_mappings(self):
        if os.path.exists(self.mapping_file):
            # print(f"Loading condition mappings from {self.mapping_file}")
            return self.load_mappings()
        else:
            # print("Creating new condition mappings.")
            mappings = self.create_mappings()
            self.save_mappings(mappings)
            return mappings
    
    def create_mappings(self):
        mappings = {
            'kingdom': {}, 'phylum': {}, 'class': {}, 
            'order': {}, 'family': {}, 'genus': {}, 
            'specific_epithet': {}
        }
        
        # Create stable mappings from conditions to integer indices
        for cat in self.categories.values():
            for key in mappings.keys():
                value = cat[key]
                if value not in mappings[key]:
                    mappings[key][value] = len(mappings[key])
        
        # Add `None` as the last index for each condition
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
            'kingdom': {}, 'phylum': {}, 'class': {}, 
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

    def map_condition(self, category, error_loaded_image):
        """Map condition values to integer indices with cascading None behavior."""
        conditions = {}
        conditions_list = []
        set_none = False
        cut_off_index = 0
        if error_loaded_image:
            set_none = True
        for key in self.condition_mappings.keys():
            if set_none or random.random() < self.condition_prob:
                conditions[key] = self.condition_mappings[key]['None']
                conditions_list.append(conditions[key])
                set_none = True  # Ensures all subsequent conditions are set to None
            else:
                value = category[key]
                conditions[key] = self.condition_mappings[key].get(value, self.condition_mappings[key]['None'])
                conditions_list.append(conditions[key])
                cut_off_index += 1
        return conditions, conditions_list, cut_off_index

    def resize_image_random_cropping(self, image, resolution):
        H, W, C = image.shape
        if W >= H:
            crop_l = (W - H) // 2 if not self.training else random.randint(0, W - H)
            crop_r = crop_l + H
            crop_t = 0
            crop_b = H
        else:
            crop_t = (H - W) // 2 if not self.training else random.randint(0, H - W)
            crop_b = crop_t + W
            crop_l = 0
            crop_r = W
        image = image[crop_t:crop_b, crop_l:crop_r]
        k = float(resolution) / min(H, W)
        img = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img, [crop_t / H, crop_b / H, crop_l / W, crop_r / W]

    def resize_image_fixed_cropping(self, image, resolution, sizes):
        H, W, C = image.shape
        crop_t_rate, crop_b_rate, crop_l_rate, crop_r_rate = sizes
        crop_t, crop_b = int(crop_t_rate * H), int(crop_b_rate * H)
        crop_l, crop_r = int(crop_l_rate * W), int(crop_r_rate * W)
        image = image[crop_t:crop_b, crop_l:crop_r]
        k = float(resolution) / min(H, W)
        img = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return img

    def __len__(self):
        return len(self.annotations)
    
    def create_name(self, category, cut_off_index):
        """Create a name by concatenating each category value with an underscore."""
        name_parts = [str(category[key]) for key in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'specific_epithet']]
        name = ""
        for i in range(len(name_parts)):
            if i < cut_off_index:
                name = name + name_parts[i] + "_"
            else:
                name = name + "None_"
        return name
    
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

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_info = self.img_info[annotation['image_id']]
        
        # Load image or fallback image if not available
        img_path = os.path.join(self.base_image_url, img_info['file_name'])
        note = "Image available"
        error_loaded_image = False
        try:
            image = self.imread(img_path)
            # image = Image.open(img_path).convert('RGB')
            # image = np.array(image)
        except (FileNotFoundError, IOError):
            image = Image.open(os.path.join(self.base_image_url, self.fallback_img_path)).convert('RGB')
            image = np.array(image)
            note = "Image not available, using fallback image"
            error_loaded_image = True

        image, crop_sizes = self.resize_image_random_cropping(image, self.img_size)
        image = (image.astype(np.float32) / 127.5) - 1.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        
        # Get category info
        category = self.categories[annotation['category_id']]

        conditions_list_name = [
            f'kingdom: {category['kingdom']}',
            f'phylum: {category['phylum']}',
            f'class: {category['class']}',
            f'order: {category['order']}',
            f'family: {category['family']}',
            f'genus: {category['genus']}',
            f'specific_epithet: {category['specific_epithet']}',
        ]
        conditions, conditions_list, cut_off_index = self.map_condition(category, error_loaded_image)
        conditions_list = torch.as_tensor(conditions_list)
        name = self.create_name(category, cut_off_index) + "__" + img_info['file_name']
        if error_loaded_image:
            name = "_1" + "__" + img_info['file_name']

        name = name.replace('/', '_')
        
        return {
            'target_image': image,
            'label': annotation['category_id'],
            'conditions': conditions,
            'conditions_list': conditions_list,
            'note': note,
            'name': name,
            'conditions_list_name': conditions_list_name
        }
