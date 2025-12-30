

"""
Multi-Dataset Tactile Loader for Vision-Language-Tactile Training

Supports multiple tactile datasets without requiring CSV files:
- Touch-and-Go (TAG)
- Feeling
- Octopi
- And more...

Auto-discovers dataset structure by scanning directories.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, List, Tuple
import json
import re
from pathlib import Path


class MultiTactileDataset(Dataset):
    """
    Multi-dataset loader that combines Touch-and-Go, Feeling, Octopi, and other datasets.
    
    Supports:
    - Touch-and-Go: paired vision-tactile data
    - Feeling: tactile images with object labels
    - Octopi: tactile sequences
    - Auto-discovery of dataset structure
    
    Args:
        dataroots: Dict mapping dataset names to their root directories
                   Example: {'tag': '/path/to/TAG', 'feeling': '/path/to/feeling', 'octopi': '/path/to/octopi'}
        mode: 'train' or 'val'
        image_size: Target image size
        use_augmentation: Whether to apply data augmentation
        max_samples_per_dataset: Limit samples per dataset (None for all)
    """
    
    def __init__(
        self,
        dataroots: Dict[str, str],
        mode: str = 'train',
        image_size: int = 224,
        use_augmentation: bool = True,
        max_samples_per_dataset: Optional[int] = None,
    ):
        self.dataroots = dataroots
        self.mode = mode
        self.image_size = image_size
        self.max_samples_per_dataset = max_samples_per_dataset
        
        # Data storage
        self.samples = []  # List of (tactile_path, dataset_name, object_id, metadata)
        self.dataset_stats = {}
        
        # Build dataset
        self._build_dataset()
        
        # Transforms
        if mode == 'train' and use_augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        print(f"\n{'='*60}")
        print(f"Multi-Tactile Dataset [{mode}]")
        print(f"{'='*60}")
        for dataset_name, stats in self.dataset_stats.items():
            print(f"{dataset_name:15s}: {stats['count']:6d} samples")
        print(f"{'Total':15s}: {len(self.samples):6d} samples")
        print(f"{'='*60}\n")
    
    def _build_dataset(self):
        """Build dataset by scanning all provided dataroots."""
        for dataset_name, dataroot in self.dataroots.items():
            if not os.path.exists(dataroot):
                print(f"Warning: {dataset_name} dataroot not found: {dataroot}")
                continue
            
            if dataset_name.lower() in ['tag', 'touchandgo', 'touch-and-go']:
                self._load_tag_dataset(dataroot, dataset_name)
            elif dataset_name.lower() in ['feeling', 'feel']:
                self._load_feeling_dataset(dataroot, dataset_name)
            elif dataset_name.lower() in ['octopi']:
                self._load_octopi_dataset(dataroot, dataset_name)
            else:
                # Generic tactile dataset loader
                self._load_generic_dataset(dataroot, dataset_name)
    
    def _load_tag_dataset(self, dataroot: str, dataset_name: str):
        """
        Load Touch-and-Go dataset.
        
        Expected structure:
        dataroot/
          ├── object1/
          │   ├── gelsight_frame/
          │   │   ├── 0000000001.jpg
          │   │   └── ...
          │   └── realsense_color/
          │       └── ...
          └── object2/
              └── ...
        """
        print(f"Loading {dataset_name} from {dataroot}...")
        count = 0
        
        if not os.path.isdir(dataroot):
            return
        
        # Scan for object folders
        for obj_folder in sorted(os.listdir(dataroot)):
            obj_path = os.path.join(dataroot, obj_folder)
            if not os.path.isdir(obj_path):
                continue
            
            # Look for tactile images
            tactile_dir = os.path.join(obj_path, 'gelsight_frame')
            if not os.path.exists(tactile_dir):
                # Try alternative names
                for alt_name in ['tactile', 'gelsight', 'gel_frame']:
                    alt_path = os.path.join(obj_path, alt_name)
                    if os.path.exists(alt_path):
                        tactile_dir = alt_path
                        break
            
            if not os.path.exists(tactile_dir):
                continue
            
            # Collect tactile images
            tactile_images = sorted([
                f for f in os.listdir(tactile_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            for img_file in tactile_images:
                tactile_path = os.path.join(tactile_dir, img_file)
                
                # Look for corresponding vision image
                vision_path = None
                vision_dir = os.path.join(obj_path, 'realsense_color')
                if os.path.exists(vision_dir):
                    vision_file = img_file  # Assume same name
                    vision_path = os.path.join(vision_dir, vision_file)
                    if not os.path.exists(vision_path):
                        vision_path = None
                
                self.samples.append({
                    'tactile_path': tactile_path,
                    'vision_path': vision_path,
                    'dataset': dataset_name,
                    'object_id': obj_folder,
                    'frame_id': img_file,
                })
                count += 1
                
                if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                    break
            
            if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                break
        
        self.dataset_stats[dataset_name] = {'count': count}
    
    def _load_feeling_dataset(self, dataroot: str, dataset_name: str):
        """
        Load Feeling dataset.
        
        Expected structure:
        dataroot/
          ├── object1/
          │   ├── 001.jpg
          │   ├── 002.jpg
          │   └── ...
          ├── object2/
          │   └── ...
          └── ...
        
        Or flat structure:
        dataroot/
          ├── object1_001.jpg
          ├── object1_002.jpg
          └── ...
        """
        print(f"Loading {dataset_name} from {dataroot}...")
        count = 0
        
        if not os.path.isdir(dataroot):
            return
        
        # Check if organized by folders or flat
        subdirs = [d for d in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, d))]
        
        if subdirs:
            # Folder-based organization
            for obj_folder in sorted(subdirs):
                obj_path = os.path.join(dataroot, obj_folder)
                
                # Collect all images in this folder
                images = sorted([
                    f for f in os.listdir(obj_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                
                for img_file in images:
                    tactile_path = os.path.join(obj_path, img_file)
                    
                    self.samples.append({
                        'tactile_path': tactile_path,
                        'vision_path': None,  # Feeling dataset doesn't have vision
                        'dataset': dataset_name,
                        'object_id': obj_folder,
                        'frame_id': img_file,
                    })
                    count += 1
                    
                    if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                        break
                
                if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                    break
        else:
            # Flat organization - group by filename pattern
            all_images = sorted([
                f for f in os.listdir(dataroot)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            for img_file in all_images:
                tactile_path = os.path.join(dataroot, img_file)
                
                # Try to extract object ID from filename
                # Common patterns: "object1_001.jpg", "obj_001.jpg", "01_005.jpg"
                match = re.match(r'([a-zA-Z0-9]+)_.*', img_file)
                object_id = match.group(1) if match else 'unknown'
                
                self.samples.append({
                    'tactile_path': tactile_path,
                    'vision_path': None,
                    'dataset': dataset_name,
                    'object_id': object_id,
                    'frame_id': img_file,
                })
                count += 1
                
                if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                    break
        
        self.dataset_stats[dataset_name] = {'count': count}
    
    def _load_octopi_dataset(self, dataroot: str, dataset_name: str):
        """
        Load Octopi dataset.
        
        Expected structure:
        dataroot/
          ├── sequence1/
          │   ├── 0000000001.jpg
          │   ├── 0000000002.jpg
          │   └── ...
          ├── sequence2/
          │   └── ...
          └── ...
        
        Or processed structure:
        dataroot/
          ├── processed/
          │   ├── seq1/
          │   │   └── ...
          │   └── seq2/
          │       └── ...
        """
        print(f"Loading {dataset_name} from {dataroot}...")
        count = 0
        
        # Check for 'processed' subdirectory
        if os.path.exists(os.path.join(dataroot, 'processed')):
            dataroot = os.path.join(dataroot, 'processed')
        
        if not os.path.isdir(dataroot):
            return
        
        # Scan for sequence folders
        for seq_folder in sorted(os.listdir(dataroot)):
            seq_path = os.path.join(dataroot, seq_folder)
            if not os.path.isdir(seq_path):
                continue
            
            # Collect images in sequence
            images = sorted([
                f for f in os.listdir(seq_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            
            for img_file in images:
                tactile_path = os.path.join(seq_path, img_file)
                
                self.samples.append({
                    'tactile_path': tactile_path,
                    'vision_path': None,  # Octopi is tactile-only
                    'dataset': dataset_name,
                    'object_id': seq_folder,
                    'frame_id': img_file,
                })
                count += 1
                
                if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                    break
            
            if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                break
        
        self.dataset_stats[dataset_name] = {'count': count}
    
    def _load_generic_dataset(self, dataroot: str, dataset_name: str):
        """
        Generic loader for unknown dataset structure.
        Recursively scans for image files.
        """
        print(f"Loading {dataset_name} from {dataroot} (generic mode)...")
        count = 0
        
        # Recursively find all images
        for root, dirs, files in os.walk(dataroot):
            for file in sorted(files):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    tactile_path = os.path.join(root, file)
                    
                    # Extract relative path as object_id
                    rel_path = os.path.relpath(root, dataroot)
                    object_id = rel_path.replace(os.sep, '_')
                    
                    self.samples.append({
                        'tactile_path': tactile_path,
                        'vision_path': None,
                        'dataset': dataset_name,
                        'object_id': object_id,
                        'frame_id': file,
                    })
                    count += 1
                    
                    if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                        break
            
            if self.max_samples_per_dataset and count >= self.max_samples_per_dataset:
                break
        
        self.dataset_stats[dataset_name] = {'count': count}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load tactile image
        try:
            tactile_img = Image.open(sample['tactile_path']).convert('RGB')
            tactile_tensor = self.transform(tactile_img)
        except Exception as e:
            print(f"Error loading tactile image {sample['tactile_path']}: {e}")
            # Return a black image as fallback
            tactile_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # Load vision image if available
        vision_tensor = None
        if sample['vision_path'] and os.path.exists(sample['vision_path']):
            try:
                vision_img = Image.open(sample['vision_path']).convert('RGB')
                # Keep as PIL for SigLIP processor
                vision_pil = vision_img
            except Exception as e:
                print(f"Error loading vision image {sample['vision_path']}: {e}")
                vision_pil = None
        else:
            vision_pil = None
        
        return {
            'tactile': tactile_tensor,
            'vision': vision_pil,  # PIL Image or None
            'dataset': sample['dataset'],
            'object_id': sample['object_id'],
            'frame_id': sample['frame_id'],
            'path': sample['tactile_path'],
        }


def collate_multi_dataset(batch):
    """
    Custom collate function for multi-dataset loader.
    
    Returns:
        Dict with:
        - tactile: Tensor (B, C, H, W)
        - vision: List[PIL.Image] or List[None]
        - dataset: List[str]
        - object_id: List[str]
        - frame_id: List[str]
        - path: List[str]
    """
    tactile = torch.stack([b['tactile'] for b in batch], dim=0)
    vision = [b['vision'] for b in batch]
    dataset = [b['dataset'] for b in batch]
    object_id = [b['object_id'] for b in batch]
    frame_id = [b['frame_id'] for b in batch]
    path = [b['path'] for b in batch]
    
    return {
        'tactile': tactile,
        'vision': vision,
        'dataset': dataset,
        'object_id': object_id,
        'frame_id': frame_id,
        'path': path,
    }


# Quick test
if __name__ == "__main__":
    # Example usage
    dataroots = {
        'tag': '/path/to/TouchAndGo/dataset',
        'feeling': '/path/to/feeling',
        'octopi': '/path/to/octopi',
    }
    
    dataset = MultiTactileDataset(
        dataroots=dataroots,
        mode='train',
        image_size=224,
        use_augmentation=True,
        max_samples_per_dataset=100,  # Limit for testing
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Tactile shape: {sample['tactile'].shape}")
        print(f"  Vision: {sample['vision']}")
        print(f"  Dataset: {sample['dataset']}")
        print(f"  Object ID: {sample['object_id']}")
        print(f"  Frame ID: {sample['frame_id']}")
