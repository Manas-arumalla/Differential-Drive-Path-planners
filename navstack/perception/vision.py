import numpy as np

class ColorDetector:
    def __init__(self):
        # Validation thresholds
        self.min_area = 20 # Minimum pixels to count as object

    def detect(self, img, target_color):
        """
        Detect blob of target_color in img (H, W, 3).
        Returns: (cx_norm, cy_norm, area_norm) or None
        cx_norm: -1.0 (Left) to 1.0 (Right)
        """
        if img is None: return None
        
        # Simple RGB Thresholds
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        
        if target_color == "Red":
            # Red: R dominant over G and B
            mask = (r > g + 30) & (r > b + 30) & (r > 60)
        elif target_color == "Green":
            mask = (g > r + 30) & (g > b + 30) & (g > 60)
        elif target_color == "Blue":
            mask = (b > r + 30) & (b > g + 30) & (b > 60)
        
        if mask is None: return None
        
        # Find pixels
        y_idxs, x_idxs = np.where(mask)
        
        if len(x_idxs) < self.min_area:
            return None
            
        # Centroid
        cx = np.mean(x_idxs)
        cy = np.mean(y_idxs)
        
        # Normalize to [-1, 1]
        h, w = img.shape[:2]
        cx_norm = (cx - (w / 2)) / (w / 2)
        cy_norm = (cy - (h / 2)) / (h / 2)
        
        # Area Fraction
        area_norm = len(x_idxs) / (w * h)
        
        return cx_norm, cy_norm, area_norm
