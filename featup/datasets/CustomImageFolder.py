import os

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


class CustomImageFolder(Dataset):

    def __init__(self, root, transform=None, **kwargs):
        self.root = root
        self.transform = transform
        self.paths = sorted([
            os.path.join(root, f) for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ])
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in {root} with extensions {IMAGE_EXTENSIONS}")

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return {"img": image, "img_path": image_path}

    def __len__(self):
        return len(self.paths)
