import os
import yaml
import numpy as np
from tqdm import tqdm
import glob


def load_config(config_path):
    """
    Loads a YAML configuration file into a Python dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist at the given path.
        yaml.YAMLError: If the file contains invalid YAML syntax.
        UnicodeDecodeError: If the file cannot be decoded with UTF-8 encoding.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_embeddings_from_disk(cache_dir, sample_limit=None):
    """
    Loads precomputed embeddings from `.npy` files in a specified directory.

    Embeddings are assumed to be stored as individual NumPy files (`.npy`) in the given cache directory.
    Files are loaded in sorted order (lexicographic by filename).
    Optionally limits the number of embeddings loaded for debugging or sampling.

    Args:
        cache_dir (str): Path to the directory containing `.npy` embedding files.
        sample_limit (int, optional): Maximum number of embeddings to load. If None, loads all available files. Default is None.

    Returns:
        np.ndarray: A 2D array of shape ``(N, D)``, where ``N`` is the number of loaded embeddings (≤ sample_limit) and ``D`` is the embedding dimension.

     Raises:
         RuntimeError: If no `.npy` files are found in the specified directory.
        ValueError: If any loaded file does not contain a 1D or 2D array compatible with stacking.
    """
    npy_files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))
    if sample_limit:
        npy_files = npy_files[:sample_limit]

    embeddings = []
    for f in tqdm(npy_files, desc="Loading embeddings from disk"):
        emb = np.load(f)
        embeddings.append(emb)

    if not embeddings:
        raise RuntimeError(f"No embeddings found in {cache_dir}")

    return np.vstack(embeddings)


def print_and_save_result(image_dir, result, quality, output_path=None, save_to_file=True):

    report_lines = [
        "=" * 100,
        f"[Visual Diversity Evaluation Result] | Dataset Path: {image_dir}",
        "=" * 100,
        f"- Number of samples: {result['n_samples']}",
        f"- Embedding dimension: {result['n_features']}",
        f"- Effective rank (normalized): {result['effective_rank_norm']:.6f}",
        f"- Participation ratio (normalized): {result['participation_ratio_norm']:.6f}",
        f"- Final diversity score (geometric mean): {result['diversity_score']:.6f}",
        f"- Dataset diversity: {quality}",
        "=" * 100,
    ]
    report_text = "\n".join(report_lines)

    print("\n" + report_text)

    if save_to_file and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\nResults saved at: {output_path}")
