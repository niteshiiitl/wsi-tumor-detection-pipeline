import torch
import numpy as np
import cv2
from torchvision import transforms
from wsi_handler import WSIPatchExtractor
from model import SimpleUNet

def run_gigapixel_inference(wsi_path: str, model_weights_path: str, patch_size: int = 512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimpleUNet(in_channels=3, out_channels=1).to(device)
    model.eval()
    
    extractor = WSIPatchExtractor(wsi_path, patch_size)
    cols, rows = extractor.get_grid_dimensions()
    
    scale_factor = 0.1
    heatmap_h = int((rows * patch_size) * scale_factor)
    heatmap_w = int((cols * patch_size) * scale_factor)
    global_heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for row in range(rows):
            for col in range(cols):
                x = col * patch_size
                y = row * patch_size
                
                patch_np = extractor.get_patch(x, y)
                
                if np.mean(patch_np) > 240: 
                    continue 
                
                patch_tensor = transform(patch_np).unsqueeze(0).to(device)
                
                pred_mask = model(patch_tensor)
                pred_mask = pred_mask.squeeze().cpu().numpy()
                
                resized_mask = cv2.resize(pred_mask, 
                                          (int(patch_size * scale_factor), int(patch_size * scale_factor)))
                
                start_y = int(row * patch_size * scale_factor)
                start_x = int(col * patch_size * scale_factor)
                
                global_heatmap[start_y:start_y+resized_mask.shape[0], 
                               start_x:start_x+resized_mask.shape[1]] = resized_mask
                
    return global_heatmap
