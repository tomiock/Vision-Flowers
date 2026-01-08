import os
import json
import numpy as np
import clip
from PIL import Image
from torch.utils.data import Dataset

class FlowersDataset(Dataset):
    def __init__(self, images_dir: str, labels_file: str, cat_json: str, preprocess, prompt_template: str = "an image of the {} flower"):
        """
        Args:
            images_dir: Path to image folder.
            labels_file: Path to .npz labels file.
            cat_json: Path to category to name json.
            preprocess: CLIP preprocessing function.
            prompt_template: String template for the prompt, must contain {}.
        """
        super().__init__()
        self.images_dir = images_dir
        self.labels = np.load(labels_file)['arr_0']
        
        # Deterministic sorting
        self.images_paths = sorted(os.listdir(images_dir)) 
        self.preprocess = preprocess
        self.prompt_template = prompt_template

        with open(cat_json) as f:
            self.cat2name = json.load(f)
            
        self.classes = [self.cat2name[str(i)] for i in range(1, len(self.cat2name) + 1)]

    def __getitem__(self, idx):
        img_name = self.images_paths[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        image_tensor = self.preprocess(image)

        label_idx = self.labels.reshape(-1)[idx] 
        class_name = self.cat2name[str(label_idx)]

        caption = self.prompt_template.format(class_name)
        
        text_token = clip.tokenize([caption], truncate=True)[0]

        return image_tensor, text_token, int(label_idx) - 1 

    def __len__(self):
        return len(self.images_paths)