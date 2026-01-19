import numpy as np


def compute_visual_diversity_score(embeddings, return_details=False):
    """
    Compute the visual diversity of a set of embeddings using a 5-step procedure.

    Args:
        embeddings: NumPy array of shape (n_samples, n_features)
        return_details: Whether to return detailed results

    Returns:
        float or dict (depending on return_details)

    Reference:
        Step 1~3: Roy & Vetterli (2007) - “The effective rank: A measure of effective dimensionality”
        Step 4: Morcos et al. (2018) - “On the importance of single directions for generalization”
        Step 5: effective rank (Steps 1~3) and participation ratio (Step 4) are combined via geometric mean to produce the final diversity score.
    """
    if embeddings.ndim != 2:
        raise ValueError("Input embeddings must be a 2-D array.")
    n_samples, n_features = embeddings.shape

    if n_samples < 2:
        raise ValueError("At least two samples are required to compute diversity.")

    """
    Step 1 - Covariance Matrix Calculation
    Purpose:
        Compute a covariance matrix to analyze how the image embedding vectors 
        are distributed or dispersed across different directions.
    Insight:
        The covariance matrix summarizes how images share or differ in visual 
        feature combinations, such as color, contrast, texture, and composition.
    """
    covariance_matrix = np.cov(embeddings, rowvar=False)

    """
    Step 2 - Eigenvalue Decomposition
    Purpose:
        Compute eigenvalues to identify the main directions (principal components)
        of variation in the covariance matrix and their corresponding magnitudes (variance).
    Insight:
        Diversity of image dataset primarily comes from these five visual axes 
        (e.g., brightness, contrast, composition, color, texture).
        Contribution of each axis to overall diversity is assessed by the magnitude of its corresponding eigenvalue.
    """
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]

    """
    Step 3 - Normalize into Probability Distribution -> Compute Entropy -> Effective Rank
    Purpose:
        Treat eigenvalues as a probability distribution to compute entropy, 
        revealing how evenly the variance is spread across directions (effective rank).
    Insight:
        If 100 images have completely different compositions, colors, and themes, entropy increases 
        -> effective rank increases. 
        If 100 images have the same background and same objects, entropy decreases 
        -> effective rank decreases.
    """
    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        # All eigenvalues are 0 -> no variance -> diversity is 0
        result = {
            "diversity_score": 0.0,
            "effective_rank_norm": 0.0,
            "participation_ratio_norm": 0.0,
            "n_samples": n_samples,
            "n_features": n_features,
        }
        if return_details:
            return result
        else:
            return 0.0

    probability = eigenvalues / total_variance
    probability = probability[probability > 0]
    entropy = -np.sum(probability * np.log(probability + 1e-12))
    effective_rank = np.exp(entropy)

    """
    Step 4 - Participation Ratio Calculation
    Purpose:
        Measure how evenly the variance is distributed -> detect extreme skewness.
    Insight:
        High participation ratio = all principal components contribute evenly = balanced diversity
        Low participation ratio = one or two axes dominate = biased diversity
    """
    participation_ratio = (total_variance ** 2) / np.sum(eigenvalues ** 2)

    """
    Step 5 - Normalize -> Geometric Mean -> Final Diversity Score
    Purpose:
        Normalize the two metrics (effective rank and participation ratio) to the range 0~1, 
        then combine them using the geometric mean -> compute the final score.
    Insight:
        Effective rank increases + Participation ratio increases -> Diversity score increases
        High effective rank but low participation ratio -> Diversity skewed toward specific directions -> Score decreases
        High participation ratio but low effective rank -> Variance is even but total variance is small -> Score decreases
    """
    effective_rank_norm = effective_rank / n_features
    participation_ratio_norm = participation_ratio / n_features
    diversity_score = np.sqrt(effective_rank_norm * participation_ratio_norm)

    if return_details:
        return {
            "diversity_score": float(diversity_score),
            "effective_rank_norm": float(effective_rank_norm),
            "participation_ratio_norm": float(participation_ratio_norm),
            "n_samples": n_samples,
            "n_features": n_features,
        }

    return float(diversity_score)


def interpret_diversity_score(score):

    thresholds = [
        (0.50, "Very Good"),
        (0.40, "Good"),
        (0.30, "Normal"),
        (0.20, "Low"),
        (0.00, "Very Low")
    ]

    for threshold, label in thresholds:
        if score >= threshold:
            return label
