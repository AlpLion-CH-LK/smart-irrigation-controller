"""Crop profiles for the irrigation controller."""

from irrigation.crops.base import CropProfile
from irrigation.crops.chili import ChiliProfile

CROP_REGISTRY: dict[str, type[CropProfile]] = {
    "chili": ChiliProfile,
}


def get_crop_profile(name: str) -> CropProfile:
    """Return an instantiated crop profile by name.

    Args:
        name: Crop identifier (e.g. ``"chili"``).

    Returns:
        An instance of the corresponding :class:`CropProfile`.

    Raises:
        ValueError: If the crop name is not registered.
    """
    key = name.lower().strip()
    if key not in CROP_REGISTRY:
        available = ", ".join(sorted(CROP_REGISTRY))
        raise ValueError(f"Unknown crop '{name}'. Available crops: {available}")
    return CROP_REGISTRY[key]()


__all__ = ["CropProfile", "ChiliProfile", "CROP_REGISTRY", "get_crop_profile"]
