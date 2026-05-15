"""
Geometric Enhancement Module for THRP Pipeline
Implements CLAHE and Canny Edge Fusion for aircraft image preprocessing
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class GeometricEnhancementModule:
    """
    Preprocessing module that applies CLAHE and Canny edge fusion
    to enhance aircraft structural features and defeat camouflage textures.
    """
    
    def __init__(self, clahe_clip_limit: float = 2.0, clahe_tile_size: int = 8):
        """
        Initialize the enhancement module.
        
        Args:
            clahe_clip_limit: Contrast limit for CLAHE (default: 2.0)
            clahe_tile_size: Tile size for CLAHE (default: 8x8)
        """
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=(clahe_tile_size, clahe_tile_size)
        )
        
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        Normalizes lighting and defeats atmospheric haze.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space for better CLAHE application
        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            l_enhanced = self.clahe.apply(l_channel)
            lab[:, :, 0] = l_enhanced
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            enhanced = self.clahe.apply(image)
            
        return enhanced
    
    def apply_canny_edge_detection(self, image: np.ndarray, 
                                   threshold1: int = 50, 
                                   threshold2: int = 150) -> np.ndarray:
        """
        Apply Canny edge detection to extract structural features.
        
        Args:
            image: Input image
            threshold1: Lower threshold for Canny (default: 50)
            threshold2: Upper threshold for Canny (default: 150)
            
        Returns:
            Binary edge map
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
        return edges
    
    def fuse_clahe_and_edges(self, image: np.ndarray, 
                             edge_weight: float = 0.3) -> np.ndarray:
        """
        Fuse CLAHE-enhanced image with Canny edge map.
        Emphasizes structural geometry and silhouettes over camouflage.
        
        Args:
            image: Input image
            edge_weight: Weight for edge map in fusion (0-1, default: 0.3)
            
        Returns:
            Fused image emphasizing structure
        """
        # Step 1: Apply CLAHE
        clahe_enhanced = self.apply_clahe(image)
        
        # Step 2: Generate edge map
        edges = self.apply_canny_edge_detection(clahe_enhanced)
        
        # Step 3: Convert edges to 3-channel for blending
        if len(clahe_enhanced.shape) == 3:
            edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            edges_3channel = edges
            
        # Normalize edges to 0-255 range
        edges_3channel = edges_3channel.astype(np.float32)
        edges_normalized = (edges_3channel / 255.0) * 255
        
        # Step 4: Fuse using weighted blending
        clahe_float = clahe_enhanced.astype(np.float32)
        fused = cv2.addWeighted(clahe_float, 1.0 - edge_weight, 
                                edges_normalized, edge_weight, 0)
        
        return np.clip(fused, 0, 255).astype(np.uint8)
    
    def process_roi(self, image: np.ndarray, 
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Process a region of interest for Stage 2 input.
        
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2) or None for full image
            
        Returns:
            Preprocessed image ready for specialist models
        """
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]
        else:
            cropped = image
            
        # Apply geometric enhancement
        enhanced = self.fuse_clahe_and_edges(cropped)
        
        return enhanced


def visualize_preprocessing(image_path: str, output_path: Optional[str] = None) -> None:
    """
    Visualize the preprocessing pipeline stages.
    
    Args:
        image_path: Path to input image
        output_path: Path to save visualization (optional)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Initialize module
    module = GeometricEnhancementModule()
    
    # Apply stages
    clahe_result = module.apply_clahe(image)
    edges_result = module.apply_canny_edge_detection(image)
    fused_result = module.fuse_clahe_and_edges(image)
    
    # Create comparison image
    h, w = image.shape[:2]
    
    # Resize edges to match original for comparison
    edges_3ch = cv2.cvtColor(edges_result, cv2.COLOR_GRAY2BGR)
    
    # Create grid
    top_row = np.hstack([image, clahe_result])
    bottom_row = np.hstack([edges_3ch, fused_result])
    comparison = np.vstack([top_row, bottom_row])
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "CLAHE Enhanced", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Canny Edges", (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Fused Result", (w + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display or save
    if output_path:
        cv2.imwrite(output_path, comparison)
        print(f"Preprocessing visualization saved to {output_path}")
    else:
        cv2.imshow("THRP Preprocessing Pipeline", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Geometric Enhancement Module loaded successfully!")
