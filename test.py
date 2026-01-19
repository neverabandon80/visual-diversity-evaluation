import os
import argparse
import traceback
from embedders.sscd_embedding import SSCDImageEmbedder
from diversity.diversity_calculation import compute_visual_diversity_score, interpret_diversity_score
from utils import load_config, load_embeddings_from_disk, print_and_save_result


def main(config):
    """
    Visual Diversity Evaluation for MLLM Datasets

    Description:
        This module implements a method similar to HuggingFace's approach for measuring
        visual diversity in the FineVision dataset. Specifically, it replicates the
        logic described in the "How diverse are the datasets?" section.

    Reference:
        HuggingFace FineVision Space - https://huggingface.co/spaces/HuggingFaceM4/FineVision

    MLLM Dataset Diversity Statistics:
        Name           Images      effective_rank_norm   participation_ratio_norm   diversity_score
        -------------------------------------------------------------------------------------------
        Cauldron       2.0M             324.05                  129.22                0.400
        LLaVa-Vision   2.5M             267.89                  87.05                 0.298
        Cambrian-7M    5.4M             359.73                  152.70                0.458
        FineVision     17.3M            359.22                  182.52                0.500
        -------------------------------------------------------------------------------------------

    Diversity Score Analysis:
        diversity_score >= 0.50        : Very Good
                                         - FineVision level
                                         - Optimal for large-scale MLLM training
        0.40 <= diversity_score < 0.50 : Good
                                         - Cambrian level
                                         - Suitable for general MLLM training
        0.30 <= diversity_score < 0.40 : Normal
                                         - LLaVa level
                                         - Filtering or data augmentation recommended
        0.20 <= diversity_score < 0.30 : Low
                                         - May contain serious bias or duplicates
        diversity_score < 0.20         : Very Low
                                         - Quality inspection required
    """

    image_dir = config["image_dir"]
    result_path = config["result_path"]
    model_path = config.get("model_path", None)
    max_images = config.get("max_images", None)
    batch_size = config.get("batch_size", 32)
    device = config.get("device", None)
    use_multi_gpu = config.get("use_multi_gpu", False)
    gpu_ids = config.get("gpu_ids", None)
    use_disk_cache = config.get("use_disk_cache", False)
    embedding_cache_dir = config["embedding_cache_dir"]
    skip_existing = config.get("skip_existing", True)

    # Step 1: Collect image paths (include subdirectories)
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_paths = []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    image_paths = sorted(image_paths)

    if max_images:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise ValueError(f"Could not find any images in '{image_dir}' or its subdirectories.")

    print(f"Found {len(image_paths)} images to process.")

    # Step 2: Extract embeddings using SSCDImageEmbedder
    embedder = SSCDImageEmbedder(model_path=model_path, device=device, use_multi_gpu=use_multi_gpu, gpu_ids=gpu_ids)

    if use_disk_cache:
        print(f"Using disk cache mode. Saving embeddings to: {embedding_cache_dir}")
        saved_npy_paths = embedder.extract_batch_to_disk(
            image_paths,
            batch_size=batch_size,
            cache_dir=embedding_cache_dir,
            skip_existing=skip_existing
        )
        print(f"Loading embeddings from disk...")
        embeddings = load_embeddings_from_disk(embedding_cache_dir)
        valid_paths = []
    else:
        print("Using in-memory mode (not recommended for large datasets)")
        embeddings, valid_paths = embedder.extract_batch(image_paths, batch_size=batch_size)

    if embeddings.size == 0:
        raise RuntimeError("No valid embeddings could be extracted.")

    print(f"Loaded embeddings shape: {embeddings.shape}")

    # Step 3: Calculate diversity score, then print and save result
    result = compute_visual_diversity_score(embeddings, return_details=True)
    quality = interpret_diversity_score(result['diversity_score'])

    print_and_save_result(image_dir, result, quality, output_path=result_path, save_to_file=True)


if __name__ == "__main__":

    """
    Configuration Options
    ---------------------

    This script can be executed with different configuration files depending on
    the desired hardware setup and caching strategy.

    Usage:
        python test.py --config <CONFIG_PATH>

    Available Configurations:
        1. Automatic Multi-GPU (without local cache):
            --config configuration/config_auto_multi_gpu.yaml

        2. Single GPU or Specific GPU (without local cache):
            --config configuration/config_specific_gpu.yaml

        3. CPU (without local cache):
            --config configuration/config_cpu.yaml

        4. Single GPU or Specific GPU (with local cache):
            --config configuration/config_specific_gpu_local_cache.yaml

        5. Automatic Multi-GPU (with local cache):
            --config configuration/config_specific_multi_gpu_local_cache.yaml
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="configuration/config_auto_multi_gpu.yaml",
                        help="Path to the YAML configuration file"
                        )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        main(config)
    except Exception as e:
        error_msg = (f"Program crashed.\n\n"
                     f"Error Type: {type(e).__name__}\n"
                     f"Error Message: {str(e)}\n\n"
                     f"Traceback:\n{traceback.format_exc()}")

        print("\n" + "=" * 80)
        print(error_msg)
        print("=" * 80 + "\n")

        crash_log_dir = "./crash_result/"
        os.makedirs(crash_log_dir, exist_ok=True)

        crash_log_path = os.path.join(crash_log_dir, "crash_log.txt")
        with open(crash_log_path, "w", encoding="utf-8") as f:
            f.write(error_msg)

        print(f"Crash log saved to: {crash_log_path}")
        print("Please check the log file for details.")

        exit(1)
