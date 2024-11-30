import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image

def apply_single_transform(img_tensor, transform_name):
    # Convert tensor to PIL image and remove normalization for better visualization
    img_pil = transforms.ToPILImage()(img_tensor * 0.3081 + 0.1307)
    
    if transform_name == "rotation":
        # Fixed angle rotation for better visualization
        return transforms.ToTensor()(transforms.functional.rotate(img_pil, angle=15))
    
    elif transform_name == "translate":
        # Fixed translation for better visualization
        return transforms.ToTensor()(transforms.functional.affine(
            img_pil,
            angle=0,
            translate=(5, 5),  # Fixed 5 pixel translation
            scale=1.0,
            shear=0
        ))
    
    elif transform_name == "scale":
        # Fixed scaling for better visualization
        return transforms.ToTensor()(transforms.functional.affine(
            img_pil,
            angle=0,
            translate=(0, 0),
            scale=1.2,  # Fixed 120% scaling
            shear=0
        ))
    
    elif transform_name == "erase":
        # Fixed position erasing for better visualization
        img_tensor = transforms.ToTensor()(img_pil)
        h, w = img_tensor.shape[1:]
        # Erase a fixed rectangle in the middle
        img_tensor[:, h//3:2*h//3, w//3:2*w//3] = 0
        return img_tensor
    
    return transforms.ToTensor()(img_pil)

def save_transformed_images(dataset, num_images=3, save_dir='data/augmented_samples'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving images to: {os.path.abspath(save_dir)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define augmentation types
    aug_types = ["rotation", "translate", "scale", "erase"]
    
    # Get some random training images
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    # Create a figure for each augmentation type
    for aug_type in aug_types:
        fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
        fig.suptitle(f'Augmentation: {aug_type.capitalize()}')
        
        for i, idx in enumerate(indices):
            # Original image
            img_orig, label = dataset[idx]
            
            # Plot original
            orig_img = img_orig * 0.3081 + 0.1307  # Denormalize for visualization
            axes[0, i].imshow(orig_img.squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f'Original ({label})')
            
            # Apply specific augmentation
            img_aug = apply_single_transform(img_orig.clone(), aug_type)
            
            # Plot augmented
            axes[1, i].imshow(img_aug.squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f'{aug_type.capitalize()}')
            
            # Save individual images
            plt.figure(figsize=(3, 3))
            plt.imshow(orig_img.squeeze(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'original_{i}_label_{label}.png'))
            plt.close()
            
            plt.figure(figsize=(3, 3))
            plt.imshow(img_aug.squeeze(), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f'aug_{aug_type}_{i}_label_{label}.png'))
            plt.close()
        
        # Save combined view for this augmentation
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'combined_{aug_type}.png'))
        plt.close()

if __name__ == "__main__":
    # Base transformations (without augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset with base transformations
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Save transformed images
    save_transformed_images(dataset)
    print("Augmentation samples generated successfully!")