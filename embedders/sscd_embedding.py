import os
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from data_loaders.custom_dataset import CustomImageDataset, collate_fn


class SSCDImageEmbedder:
    """
    Extracts image embeddings using the Self-Supervised Copy Detection (SSCD) model.

    This class loads a pre-trained TorchScript SSCD model and provides utilities to
    extract fixed-dimensional embeddings from batches of images. It supports CPU,
    single-GPU, and multi-GPU (via DataParallel) inference, and includes options to
    cache embeddings to disk.

    References:
        - Pizzi et al. (2022), "A Self-Supervised Descriptor for Image Copy Detection"
        - Official implementation (Facebook):https://github.com/facebookresearch/sscd-copy-detection
        - Reference implementation (HuggingFace): https://github.com/huggingface/large-scale-image-deduplication/blob/main/compute_embeddings.py
    """
    def __init__(self, model_path="./models/sscd_disc_mixup.torchscript.pt", device=None, use_multi_gpu=False, gpu_ids=None):
        """
        Initializes the SSCD embedder with the specified model and device configuration.

        Args:
            model_path (str): Path to the TorchScript SSCD model file. Default is "./models/sscd_disc_mixup.torchscript.pt".
            device (str, optional): Device to run inference on. Accepts "cpu", "cuda", or "cuda:X" (e.g., "cuda:0"). If None, uses CUDA if available, otherwise CPU.
            use_multi_gpu (bool): Whether to enable multi-GPU inference using ``torch.nn.DataParallel``. Only effective if multiple CUDA devices are available.
            gpu_ids (list of int, optional): Specific GPU IDs to use for multi-GPU mode. If None, uses all available GPUs. Ignored if ``use_multi_gpu=False``.

        Raises:
            RuntimeError: If CUDA is requested but not available.
            ValueError: If an invalid device string or GPU ID is provided.
            FileNotFoundError: If the model file does not exist at ``model_path``.
            RuntimeError: If the TorchScript model fails to load.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available, but device was set to CUDA.")
                if ":" in device:
                    gpu_id = int(device.split(":")[1])
                    if gpu_id >= torch.cuda.device_count():
                        raise ValueError(f"Invalid GPU device: {device}. Only {torch.cuda.device_count()} GPUs available.")
            elif device != "cpu":
                raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda[:X]'.")

            self.device = device

        print(f"Loading TorchScript SSCD model ({self.device.upper()})")

        try:
            self.model = torch.jit.load(model_path).eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to load TorchScript model: {e}")

        self.use_multi_gpu = use_multi_gpu and "cuda" in str(self.device) and torch.cuda.device_count() > 1
        if self.use_multi_gpu:
            if gpu_ids is not None:
                if not isinstance(gpu_ids, (list, tuple)) or len(gpu_ids) == 0:
                    raise ValueError("gpu_ids must be a non-empty list or tuple of integers")
                if max(gpu_ids) >= torch.cuda.device_count():
                    raise ValueError(f"Invalid GPU ID in {gpu_ids}. Available GPUs: 0 ~ {torch.cuda.device_count() - 1}")
                device_ids = list(gpu_ids)
            else:
                device_ids = list(range(torch.cuda.device_count()))

            print(f"Using DataParallel on GPUs: {device_ids}")
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.device = f"cuda:{device_ids[0]}"

        self.model = self.model.to(self.device)

        self.transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("SSCD Image Embedder is ready.")

    def extract_batch(self, image_paths, batch_size=32):
        """
        Extracts SSCD embeddings for a list of image paths in batched mode.
        Invalid or unreadable images are automatically skipped (thanks to the custom ``collate_fn``).
        The returned embeddings correspond only to successfully processed images.

        Args:
            image_paths (list of str): List of absolute or relative paths to image files.
            batch_size (int): Number of images to process per batch. Default is 32.

        Returns:
            tuple:
                - embeddings (np.ndarray): A 2D NumPy array of shape ``(N, D)``, where ``N`` is the number of successfully processed images and ``D`` is the embedding dimension (e.g., 2048 for SSCD).
                - valid_paths (list of str): List of image paths that were successfully processed, in the same order as the embeddings.

        Note:
            This method loads all embeddings into memory. For very large datasets, consider using ``extract_batch_to_disk`` to avoid memory overflow.
        """
        dataset = CustomImageDataset(image_paths, transform=self.transform)

        num_workers = 2

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            prefetch_factor=None,
            persistent_workers=False
        )

        all_embeddings = []
        all_valid_paths = []

        with torch.no_grad():
            for batch_tensor, batch_paths in tqdm(dataloader, desc="SSCD Embedding Extraction", total=len(dataloader)):
                if batch_tensor is None:
                    continue
                batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                embeddings = self.model(batch_tensor)

                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]

                all_embeddings.append(embeddings.cpu().numpy())
                all_valid_paths.extend(batch_paths)

        if not all_embeddings:
            return np.array([]), []

        embeddings = np.vstack(all_embeddings)

        return embeddings, all_valid_paths

    def extract_batch_to_disk(self, image_paths, batch_size=32, cache_dir="./embedding_cache/", skip_existing=True):
        """
        Extracts embeddings and saves them individually to disk as `.npy` files.
        Each embedding is saved under a filename derived from the hash of its image path,
        enabling deterministic caching. Optionally skips already cached embeddings.

        Args:
            image_paths (list of str): List of image file paths.
            batch_size (int): Number of images per batch. Default is 32.
            cache_dir (str): Directory where embedding files will be saved. Created if it doesn't exist. Default is "./embedding_cache/".
            skip_existing (bool): If True, skips processing images whose embeddings already exist in ``cache_dir``. Default is True.

        Returns:
            list of str: List of full paths to the saved `.npy` embedding files, one per successfully processed image (in the same order as input, excluding failed images).

        Note:
            Embedding filenames are generated as:
            ``emb_{abs(hash(img_path)) % (10**16)}.npy``.
            This avoids filename collisions while keeping names deterministic.
        """
        os.makedirs(cache_dir, exist_ok=True)
        dataset = CustomImageDataset(image_paths, transform=self.transform)

        num_workers = 2

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            prefetch_factor=None,
            persistent_workers=False
        )

        saved_paths = []
        total_processed = 0

        with torch.no_grad():
            for batch_idx, (batch_tensor, batch_paths) in enumerate(
                    tqdm(dataloader, desc="SSCD Embedding Extraction", total=len(dataloader))):
                if batch_tensor is None:
                    continue

                if skip_existing:
                    batch_saved = True
                    batch_npy_paths = []
                    for img_path in batch_paths:
                        filename = f"emb_{abs(hash(img_path)) % (10 ** 16)}.npy"
                        npy_path = os.path.join(cache_dir, filename)
                        batch_npy_paths.append(npy_path)
                        if not os.path.exists(npy_path):
                            batch_saved = False
                    if batch_saved:
                        saved_paths.extend(batch_npy_paths)
                        continue

                batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                embeddings = self.model(batch_tensor)

                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]

                embeddings = embeddings.cpu().numpy()

                for i, img_path in enumerate(batch_paths):
                    filename = f"emb_{abs(hash(img_path)) % (10 ** 16)}.npy"
                    save_path = os.path.join(cache_dir, filename)
                    np.save(save_path, embeddings[i])
                    saved_paths.append(save_path)
                    total_processed += 1

        print(f"Saved {total_processed} new embeddings to {cache_dir}")
        print(f"Total embedding files ready: {len(saved_paths)}")

        return saved_paths
