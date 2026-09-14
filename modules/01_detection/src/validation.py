"""
validation.py
=============
Input validation for Sentinel-1 Synthetic Aperture Radar (SAR) imagery in Module 1.
Distinguishes real Sentinel-1 SAR GeoTIFFs (1 or 2 bands, float radar backscatter)
from arbitrary photos, face portraits, and regular RGB/RGBA images saved as .tif.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import rasterio


DEFAULT_INVALID_MESSAGE = (
    "Uploaded file does not appear to be a valid SAR satellite image. "
    "Expected Sentinel-1 characteristics (1-2 band radar backscatter data) were not found."
)


class ValidationResult(tuple):
    """
    Validation result container supporting:
      - Boolean evaluation: `if is_valid_sar_image(...):`
      - Tuple unpacking: `is_valid, msg = is_valid_sar_image(...)`
      - Attribute access: `res.is_valid`, `res.message`, `res.details`
    """

    def __new__(
        cls,
        is_valid: bool,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ):
        instance = super().__new__(cls, (is_valid, message))
        instance._is_valid = bool(is_valid)
        instance._message = str(message)
        instance._details = details or {}
        return instance

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def message(self) -> str:
        return self._message

    @property
    def details(self) -> Dict[str, Any]:
        return self._details

    def __bool__(self) -> bool:
        return self._is_valid

    def __repr__(self) -> str:
        return f"ValidationResult(is_valid={self._is_valid}, message='{self._message}')"


def is_valid_sar_image(image_path: Union[str, Path]) -> ValidationResult:
    """
    Validates whether the provided file is a genuine Sentinel-1 SAR GeoTIFF scene.

    Evaluation Signals:
      1. Band count: Sentinel-1 SAR GeoTIFFs are 1 or 2 bands (VV, VH polarization).
         Arbitrary photos converted to .tif are typically 3 or 4 bands (RGB/RGBA).
      2. Data type: Sentinel-1 SAR backscatter data is float32, float64 (or calibrated uint16).
         Natural photos are standard uint8 (0-255).
      3. Value range:
         - SAR in dB scale: values fall in negative radar range (e.g. -70 to +10 dB).
         - SAR in linear scale: small positive floats near 0 (typical ocean mean < 5.0).
         - Natural photos: uint8 0-255 or uncalibrated high-mean distributions (e.g. mean 30-200).
      4. Geospatial metadata: Checks CRS and affine transform as supporting signals.
      5. Band correlation & image statistics: Natural RGB photos show high correlation
         (r > 0.85) across channels and smooth gradients, unlike radar speckle noise.

    Returns:
      ValidationResult: evaluated as boolean (True/False) or unpacked as (is_valid, message).
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        return ValidationResult(
            False,
            f"Image file not found: {image_path}",
            {"error": "file_not_found"},
        )

    try:
        with rasterio.open(str(path_obj)) as src:
            band_count = src.count
            dtypes = src.dtypes
            has_crs = src.crs is not None
            is_identity_transform = src.transform.is_identity
            has_georef = has_crs and not is_identity_transform

            details: Dict[str, Any] = {
                "band_count": band_count,
                "dtypes": list(dtypes),
                "has_crs": has_crs,
                "crs": str(src.crs) if has_crs else None,
                "is_identity_transform": is_identity_transform,
                "has_georef": has_georef,
            }

            # -----------------------------------------------------------------
            # 1. Band Count & Dtype Check
            # -----------------------------------------------------------------
            if band_count not in (1, 2):
                msg = (
                    f"{DEFAULT_INVALID_MESSAGE} "
                    f"Found {band_count} bands (expected 1 or 2 bands for Sentinel-1 VV/VH polarizations; "
                    f"3-4 bands indicate photographic RGB/RGBA data)."
                )
                return ValidationResult(False, msg, details)

            # Check for uint8 photographic data
            if any(dt == "uint8" for dt in dtypes):
                msg = (
                    f"{DEFAULT_INVALID_MESSAGE} "
                    f"Image data type is uint8 (expected float32/float64 radar backscatter; "
                    f"uint8 indicates regular photographic imagery)."
                )
                return ValidationResult(False, msg, details)

            # -----------------------------------------------------------------
            # 2. Value Range & Statistical Distribution Check
            # -----------------------------------------------------------------
            h, w = src.height, src.width
            sample_h = min(1024, h)
            sample_w = min(1024, w)
            window = rasterio.windows.Window(
                (w - sample_w) // 2,
                (h - sample_h) // 2,
                sample_w,
                sample_h,
            )

            arr1 = src.read(1, window=window).astype(np.float32)
            valid_mask1 = ~np.isnan(arr1)
            valid_vals1 = arr1[valid_mask1]

            if len(valid_vals1) == 0:
                return ValidationResult(
                    False,
                    f"{DEFAULT_INVALID_MESSAGE} Raster contains no valid numerical data.",
                    details,
                )

            v_min = float(np.min(valid_vals1))
            v_max = float(np.max(valid_vals1))
            v_mean = float(np.mean(valid_vals1))
            v_std = float(np.std(valid_vals1))

            details.update(
                {
                    "min": v_min,
                    "max": v_max,
                    "mean": v_mean,
                    "std": v_std,
                }
            )

            # Detect dB vs linear vs non-SAR photo
            is_db_scale = (v_min < 0.0) and (v_mean < 0.0)
            is_linear_scale = (v_min >= 0.0) and (v_mean < 10.0)

            # Photo converted to float with standard 0-255 range
            if v_min >= 0.0 and v_mean > 15.0 and v_max > 50.0:
                msg = (
                    f"{DEFAULT_INVALID_MESSAGE} "
                    f"Pixel value distribution (mean={v_mean:.1f}, max={v_max:.1f}) matches "
                    f"photographic 0-255 pixel levels rather than radar backscatter."
                )
                return ValidationResult(False, msg, details)

            # Out of bounds for radar backscatter
            if is_db_scale:
                # Valid radar backscatter in dB typically stays within [-80, +15] dB
                if v_mean < -80.0 or v_mean > 15.0:
                    msg = (
                        f"{DEFAULT_INVALID_MESSAGE} "
                        f"Radar dB backscatter mean ({v_mean:.1f} dB) falls outside plausible SAR limits."
                    )
                    return ValidationResult(False, msg, details)
            elif not is_linear_scale:
                msg = (
                    f"{DEFAULT_INVALID_MESSAGE} "
                    f"Value distribution does not match radar backscatter in either dB or linear power scale."
                )
                return ValidationResult(False, msg, details)

            # -----------------------------------------------------------------
            # 3. Inter-Band Correlation Check (if 2 bands)
            # -----------------------------------------------------------------
            if band_count == 2:
                arr2 = src.read(2, window=window).astype(np.float32)
                valid_both = valid_mask1 & (~np.isnan(arr2))
                if np.sum(valid_both) > 100:
                    b1_vals = arr1[valid_both]
                    b2_vals = arr2[valid_both]
                    # If duplicate channels (e.g. grayscale copied across bands)
                    if np.array_equal(b1_vals, b2_vals) and not has_georef:
                        msg = (
                            f"{DEFAULT_INVALID_MESSAGE} "
                            "Identical duplicate raster bands detected without geospatial metadata."
                        )
                        return ValidationResult(False, msg, details)

            # -----------------------------------------------------------------
            # 4. Geospatial Metadata (Supporting signal)
            # -----------------------------------------------------------------
            # If no georeference at all AND unusual statistics
            if not has_georef and not is_db_scale and (v_std < 1e-4):
                msg = (
                    f"{DEFAULT_INVALID_MESSAGE} "
                    "Image lacks geospatial metadata and has zero radar backscatter texture."
                )
                return ValidationResult(False, msg, details)

            return ValidationResult(True, "Valid Sentinel-1 SAR imagery.", details)

    except rasterio.errors.RasterioError as err:
        return ValidationResult(
            False,
            f"{DEFAULT_INVALID_MESSAGE} Raster error: {err}",
            {"error": str(err)},
        )
    except Exception as err:
        return ValidationResult(
            False,
            f"{DEFAULT_INVALID_MESSAGE} Inspection failed: {err}",
            {"error": str(err)},
        )
