import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from multiprocessing import Pool
from tqdm import tqdm
import os

class NoiseDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def calculate_contour_noise(self, img_path):
        """Traditional CV method (matches Eq.1 in manuscript)"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for cnt in contours if cv2.contourArea(cnt) < 50)

    def calculate_deep_noise(self, img_path, model):
        """PyTorch-based deep noise detection"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstruction = model(img_tensor)
            loss = F.mse_loss(reconstruction, img_tensor).item()
        
        return loss

    def batch_process(self, img_dir, model=None, threshold=3500, use_deep=False):
        """Hybrid noise detection with progress tracking"""
        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                    if f.endswith(('.png', '.jpg'))]
        
        noisy_paths = []
        
        with Pool() as pool:
            if use_deep:
                results = list(tqdm(pool.imap(
                    lambda x: (x, self.calculate_deep_noise(x, model)),
                    img_paths
                ), total=len(img_paths), desc="Deep Noise Analysis"))
            else:
                results = list(tqdm(pool.imap(
                    lambda x: (x, self.calculate_contour_noise(x)),
                    img_paths
                ), total=len(img_paths), desc="Contour Noise Analysis"))
            
            for path, score in results:
                if score > threshold:
                    noisy_paths.append(path)
        
        return noisy_paths

    def visualize_noise(self, img_path, save_path=None):
        """Visualization matching Figure 3 in manuscript"""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find and draw small contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 50:
                cv2.drawContours(output, [cnt], -1, (0,0,255), 1)
        
        if save_path:
            cv2.imwrite(save_path, output)
        
        return output

if __name__ == "__main__":
    # Example usage matching manuscript experiments
    detector = NoiseDetector()
    
    # Traditional method (Section 2.2)
    noisy_images = detector.batch_process(
        "data/raw",
        threshold=3500,
        use_deep=False
    )
    
    # Visualization for manuscript figures
    detector.visualize_noise(
        "data/raw/AT-0001-031-003679-R.dw.png",
        "reports/noise_visualization.png"
    )