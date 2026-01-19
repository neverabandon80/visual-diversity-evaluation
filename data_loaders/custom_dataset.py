import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    """
    A custom PyTorch Dataset class that loads images from a list of file paths.

    This dataset loads images one by one from the provided list of image paths,
    converts them to RGB mode, and applies an optional transform. If an image
    fails to load, it returns ``None`` for that sample, which can later be
    filtered out by a custom ``collate_fn``.

    Attributes:
        image_paths (list of str): List of file paths to the images.
        transform (callable, optional): A function/transform to apply to the loaded image
        (e.g., from torchvision.transforms). Defaults to None.
    """
    def __init__(self, image_paths, transform=None):
        """
        Initializes the CustomImageDataset.

        Args:
            image_paths (list of str): List of image file paths.
            transform (callable, optional): Transform to be applied on the image. Defaults to None.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of image paths in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and its path at the given index.
        The image is opened in RGB mode. If a transform is specified, it is applied to the image.
        If loading fails for any reason, a warning is printed and ``(None, img_path)`` is returned.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
             tuple: A tuple containing:
                - image (torch.Tensor or None): The transformed image tensor, or None if loading failed.
                - img_path (str): The file path of the image.
        """
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Warning: Failed to load '{img_path}': {e}")
            return None, img_path


def collate_fn(batch):
    """
    Custom collate function for use with DataLoader.

    Args:
        batch (list of tuples): Each element is a tuple ``(image, path)``, where ``image`` is a torch.Tensor or None, and ``path`` is a string.

    Returns:
         tuple: A tuple containing:
            - images (torch.Tensor or None): A stacked tensor of shape [B, C, H, W]
              containing all valid images in the batch. None if no valid images exist.
            - paths (list of str): List of file paths corresponding to the valid images.
              Empty list if no valid images exist.
    """
    filtered_batch = []

    for b in batch:
        image, path = b
        if image is not None:
            filtered_batch.append(b)

    batch = filtered_batch

    if len(batch) == 0:
        return None, []
    images, paths = zip(*batch)

    return torch.stack(images), list(paths)
