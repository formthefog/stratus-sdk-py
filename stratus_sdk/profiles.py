"""
M-JEPA-G compression profiles.

Optimized quantization for 512-dim and 768-dim M-JEPA-G embeddings.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class CompressionLevel(str, Enum):
    """Compression quality levels."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "VeryHigh"


@dataclass
class MJepaProfile:
    """M-JEPA-G compression profile."""

    dimensions: int
    precision_map: np.ndarray
    description: str
    quality: float
    ratio: float


# 768-dim profiles (medium model)

MJEPA_768_HIGH_QUALITY = MJepaProfile(
    dimensions=768,
    description="M-JEPA-G 768-dim - High Quality (99.9%+)",
    quality=99.9,
    ratio=15.2,
    precision_map=np.array(
        [8] * 192 + [8] * 192 + [6] * 192 + [4] * 192,  # First 25%  # Second 25%  # Third 25%
        dtype=np.uint8,  # Fourth 25%
    ),
)

MJEPA_768_BALANCED = MJepaProfile(
    dimensions=768,
    description="M-JEPA-G 768-dim - Balanced (99.7%+)",
    quality=99.7,
    ratio=16.8,
    precision_map=np.array(
        [8] * 192 + [6] * 192 + [6] * 192 + [4] * 192, dtype=np.uint8
    ),
)

MJEPA_768_HIGH_COMPRESSION = MJepaProfile(
    dimensions=768,
    description="M-JEPA-G 768-dim - High Compression (99.5%+)",
    quality=99.5,
    ratio=18.5,
    precision_map=np.array(
        [6] * 192 + [6] * 192 + [4] * 192 + [4] * 192, dtype=np.uint8
    ),
)

MJEPA_768_ULTRA_COMPRESSION = MJepaProfile(
    dimensions=768,
    description="M-JEPA-G 768-dim - Ultra (99.0%+)",
    quality=99.0,
    ratio=20.0,
    precision_map=np.array(
        [6] * 192 + [4] * 192 + [4] * 192 + [3] * 192, dtype=np.uint8
    ),
)

# 512-dim profiles (small model)

MJEPA_512_HIGH_QUALITY = MJepaProfile(
    dimensions=512,
    description="M-JEPA-G 512-dim - High Quality (99.9%+)",
    quality=99.9,
    ratio=15.2,
    precision_map=np.array(
        [8] * 128 + [8] * 128 + [6] * 128 + [4] * 128, dtype=np.uint8
    ),
)

MJEPA_512_BALANCED = MJepaProfile(
    dimensions=512,
    description="M-JEPA-G 512-dim - Balanced (99.7%+)",
    quality=99.7,
    ratio=16.8,
    precision_map=np.array(
        [8] * 128 + [6] * 128 + [6] * 128 + [4] * 128, dtype=np.uint8
    ),
)

MJEPA_512_HIGH_COMPRESSION = MJepaProfile(
    dimensions=512,
    description="M-JEPA-G 512-dim - High Compression (99.5%+)",
    quality=99.5,
    ratio=18.5,
    precision_map=np.array(
        [6] * 128 + [6] * 128 + [4] * 128 + [4] * 128, dtype=np.uint8
    ),
)

MJEPA_512_ULTRA_COMPRESSION = MJepaProfile(
    dimensions=512,
    description="M-JEPA-G 512-dim - Ultra (99.0%+)",
    quality=99.0,
    ratio=20.0,
    precision_map=np.array(
        [6] * 128 + [4] * 128 + [4] * 128 + [3] * 128, dtype=np.uint8
    ),
)


def get_mjepa_profile(level: CompressionLevel, dimensions: int = 768) -> MJepaProfile:
    """
    Get M-JEPA-G profile by compression level and dimension count.

    Args:
        level: Compression quality level
        dimensions: Embedding dimensions (512 or 768)

    Returns:
        MJepaProfile for the specified configuration

    Raises:
        ValueError: If dimensions is not 512 or 768
    """
    if dimensions == 512:
        profiles = {
            CompressionLevel.LOW: MJEPA_512_HIGH_QUALITY,
            CompressionLevel.MEDIUM: MJEPA_512_BALANCED,
            CompressionLevel.HIGH: MJEPA_512_HIGH_COMPRESSION,
            CompressionLevel.VERY_HIGH: MJEPA_512_ULTRA_COMPRESSION,
        }
    elif dimensions == 768:
        profiles = {
            CompressionLevel.LOW: MJEPA_768_HIGH_QUALITY,
            CompressionLevel.MEDIUM: MJEPA_768_BALANCED,
            CompressionLevel.HIGH: MJEPA_768_HIGH_COMPRESSION,
            CompressionLevel.VERY_HIGH: MJEPA_768_ULTRA_COMPRESSION,
        }
    else:
        raise ValueError(f"Unsupported dimensions: {dimensions}. Must be 512 or 768.")

    return profiles[level]


def is_mjepa_embedding(embedding: np.ndarray) -> bool:
    """
    Check if embedding matches M-JEPA-G dimensions.

    Args:
        embedding: Embedding vector

    Returns:
        True if dimensions match M-JEPA-G (512 or 768)
    """
    return embedding.shape[0] in (512, 768)


def detect_mjepa(embedding: np.ndarray) -> bool:
    """
    Auto-detect if embedding is from M-JEPA-G based on characteristics.

    Heuristics:
    - Length is 512 or 768
    - L2 norm is close to 1.0 (normalized)
    - Value distribution characteristics

    Args:
        embedding: Embedding vector

    Returns:
        True if likely M-JEPA-G embedding
    """
    if not is_mjepa_embedding(embedding):
        return False

    # Check L2 norm (should be ~1.0 for M-JEPA-G)
    norm = np.linalg.norm(embedding)
    if abs(norm - 1.0) > 0.1:
        return False

    # Check value range (continuous semantic space)
    val_range = embedding.max() - embedding.min()
    if val_range < 0.5 or val_range > 5.0:
        return False

    return True
