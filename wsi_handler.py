import openslide
import numpy as np
from PIL import Image

class WSIPatchExtractor:
    def __init__(self, wsi_path: str, patch_size: int = 512):
        self.wsi = openslide.OpenSlide(wsi_path)
        self.patch_size = patch_size
        self.width, self.height = self.wsi.level_dimensions[0]
        
    def get_patch(self, x: int, y: int) -> np.ndarray:
        patch = self.wsi.read_region((x, y), 0, (self.patch_size, self.patch_size))
        patch_rgb = patch.convert("RGB")
        return np.array(patch_rgb)

    def get_grid_dimensions(self):
        cols = self.width // self.patch_size
        rows = self.height // self.patch_size
        return cols, rows
