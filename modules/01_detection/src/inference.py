# inference.py
import os
import sys
from pathlib import Path
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import argparse
import json
from datetime import datetime, timezone
import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import torch

from model import UNet, load_spill_model

MAX_PLAUSIBLE_AREA_KM2 = 500.0  # Sanity check threshold for single-spill detections


def detect_spill(
    image_path,
    model_path="unet_spill_best.pth",
    image_size=128,
    tile_size=2048,
    input_scale="linear",
    min_area_km2=0.2,
    use_land_mask=True,
    coastal_buffer_km=1.0,
    save_json_path=None
):
    """
    Run oil spill detection on a GeoTIFF scene.
    Supports both localized patches (<=2048x2048) and full Sentinel-1 GRD swaths (via sliding window).
    Includes post-processing filters:
      - Coastal/Land mask with buffer to exclude false positives along shores and wetlands
      - Minimum area threshold to suppress micro-speckle false alarms
    """
    model = load_spill_model(model_path, device="cpu")

    with rasterio.open(image_path) as src:
        orig_h, orig_w = src.height, src.width
        has_real_crs = src.crs is not None and not src.transform.is_identity

        if has_real_crs:
            transform = src.transform
            georeference_status = "REAL"
        else:
            # Placeholder geotransform centered at lat 19.0760, lon 72.8777 (Mumbai coast)
            pixel_scale_deg = 0.0001  # Sentinel-1 ~10m resolution in degrees
            west = 72.8777 - (orig_w / 2.0) * pixel_scale_deg
            north = 19.0760 + (orig_h / 2.0) * pixel_scale_deg
            transform = from_origin(west, north, pixel_scale_deg, pixel_scale_deg)
            georeference_status = "SYNTHETIC_PLACEHOLDER"

        # Check if sliding-window tiling is needed
        is_large_swath = orig_h > tile_size or orig_w > tile_size

        if not is_large_swath:
            # Standard single-scene processing (e.g. 2048x2048 Zenodo patch)
            img = src.read(1).astype(np.float32)
            pixel_res = src.res[0] if has_real_crs else 0.0001
            result = _process_single_tile(
                img, model, transform, image_size, input_scale, georeference_status,
                min_area_km2=min_area_km2, use_land_mask=use_land_mask,
                coastal_buffer_km=coastal_buffer_km, pixel_res_deg=pixel_res
            )
        else:
            # Large swath processing (sliding window)
            result = _process_sliding_window(
                src, model, transform, orig_h, orig_w, tile_size, image_size,
                input_scale, georeference_status, min_area_km2=min_area_km2,
                use_land_mask=use_land_mask, coastal_buffer_km=coastal_buffer_km
            )

    # Sanity-check threshold check: verify largest single slick does not exceed plausible limit
    single_slick_max_km2 = 0.0
    if result["slick_polygon"] and result["slick_polygon"].get("type") in ("Polygon", "MultiPolygon"):
        poly_geom = shape(result["slick_polygon"])
        pixel_res_m = 10.0
        if poly_geom.geom_type == "Polygon":
            single_slick_max_km2 = result["area_km2"]
        elif poly_geom.geom_type == "MultiPolygon":
            pixel_res_deg = src.res[0] if has_real_crs else 0.0001
            single_slick_max_km2 = max(
                (p.area / (pixel_res_deg ** 2)) * (pixel_res_m ** 2) / 1_000_000
                for p in poly_geom.geoms
            )
            single_slick_max_km2 = round(float(single_slick_max_km2), 2)

    if single_slick_max_km2 > MAX_PLAUSIBLE_AREA_KM2:
        print(
            f"WARNING: Largest single detected slick ({single_slick_max_km2} km²) exceeds "
            f"plausible threshold ({MAX_PLAUSIBLE_AREA_KM2} km²). Refusing to save output handoff file."
        )
    elif save_json_path:
        with open(save_json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved handoff output to {save_json_path}")

    return result


def _preprocess_tile(tile, input_scale="linear"):
    """
    Applies linear-to-dB conversion (if linear) and scales to [0, 1] for U-Net input.
    Masks nodata pixels to prevent false-positive detections.
    """
    if input_scale == "db":
        # Data already in dB (Zenodo benchmark scenes)
        valid_mask = ~np.isnan(tile)
        # Fixed SAR dB clipping [-35.0 dB, -5.0 dB] (Level 1 Fix)
        vmin, vmax = -35.0, -5.0
        clipped = np.clip(tile, vmin, vmax)
        norm = (clipped - vmin) / (vmax - vmin)
        norm[~valid_mask] = 0.0
        return norm.astype(np.float32), valid_mask

    # Input is in linear power/amplitude (e.g. SNAP GeoTIFF exports)
    valid_mask = (tile > 0) & (~np.isnan(tile))
    norm = np.full_like(tile, 0.5, dtype=np.float32)  # Neutral ocean background for nodata
    if np.any(valid_mask):
        db = 10.0 * np.log10(np.maximum(tile[valid_mask], 1e-6))
        # Fixed SAR dB clipping matching Level 1 training distribution [-35 dB, -5 dB]
        vmin, vmax = -35.0, -5.0
        norm[valid_mask] = np.clip((db - vmin) / (vmax - vmin), 0.0, 1.0)

    return norm, valid_mask


def _load_land_mask(land_geojson_path="gulf_coast_land.geojson", buffer_km=1.0):
    """
    Loads land geometry and applies a coastal buffer outward into the ocean (in kilometers).
    Uses local GeoJSON if available, and falls back to fetching Natural Earth 10m online.
    """
    import os
    land_geom = None
    if not os.path.exists(land_geojson_path):
        pkg_cand = os.path.join(os.path.dirname(__file__), land_geojson_path)
        if os.path.exists(pkg_cand):
            land_geojson_path = pkg_cand

    if os.path.exists(land_geojson_path):
        try:
            with open(land_geojson_path, "r") as f:
                land_geom = shape(json.load(f))
        except Exception as e:
            print(f"Warning: Could not read local land mask {land_geojson_path}: {e}")

    if land_geom is None:
        try:
            import requests
            url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                ne_data = r.json()
                from shapely.geometry import box
                bbox = box(-92.0, 27.0, -87.0, 31.0)
                intersecting = [
                    shape(feat["geometry"]).intersection(bbox)
                    for feat in ne_data["features"]
                    if shape(feat["geometry"]).intersects(bbox)
                ]
                if intersecting:
                    land_geom = unary_union(intersecting)
                    with open(land_geojson_path, "w") as f:
                        json.dump(mapping(land_geom), f)
        except Exception as e:
            print(f"Warning: Could not fetch Natural Earth land mask: {e}")
            return None

    if land_geom is not None and buffer_km > 0:
        # Approximate degrees per km in Gulf of Mexico (lat ~29 deg: ~105 km per deg)
        deg_buf = buffer_km / 105.0
        land_geom = land_geom.buffer(deg_buf)

    return land_geom


def _filter_detections(polygons, pixel_res_deg, pixel_size_m=10.0, min_area_km2=0.2, land_buffered=None):
    """
    Applies land/coastal masking and minimum area filtering to candidate polygons.
    """
    if not polygons:
        return [], 0.0, 0

    surviving = []
    for p in polygons:
        if not p.is_valid or p.is_empty:
            continue

        # 1. Coastal / Land Mask exclusion
        if land_buffered is not None and p.intersects(land_buffered):
            diff = p.difference(land_buffered)
            if diff.is_empty:
                continue
            candidates = list(diff.geoms) if diff.geom_type in ("MultiPolygon", "GeometryCollection") else [diff]
        else:
            candidates = [p]

        # 2. Minimum Area Filter
        for c in candidates:
            if c.geom_type == "Polygon" and not c.is_empty:
                area_px = c.area / (pixel_res_deg ** 2)
                area_km2 = (area_px * (pixel_size_m ** 2)) / 1_000_000.0
                if area_km2 >= min_area_km2:
                    surviving.append((c, area_km2, area_px))

    if not surviving:
        return [], 0.0, 0

    clean_polys = [item[0] for item in surviving]
    total_area_km2 = sum(item[1] for item in surviving)
    total_pixels = int(sum(item[2] for item in surviving))

    return clean_polys, total_area_km2, total_pixels


def _process_single_tile(
    img, model, transform, image_size, input_scale, georeference_status,
    min_area_km2=0.2, use_land_mask=True, coastal_buffer_km=1.0, pixel_res_deg=0.0001
):
    """Processes a single <=2048x2048 scene."""
    orig_shape = img.shape
    norm, valid_mask = _preprocess_tile(img, input_scale=input_scale)

    img_resized = cv2.resize(norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    detection_timestamp = datetime.now(timezone.utc).isoformat()

    with torch.no_grad():
        pred = model(img_tensor)
        prob_map = torch.sigmoid(pred).squeeze().numpy()

    # Upsample mask back to original resolution using nearest-neighbor
    pred_mask_128 = (prob_map > 0.70).astype(np.uint8)  # threshold=0.70 (sweep-optimised)
    pred_mask_full = cv2.resize(
        pred_mask_128, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST
    )
    pred_mask_full = pred_mask_full & valid_mask.astype(np.uint8)

    raw_polys = []
    if np.any(pred_mask_full):
        for geom, value in shapes(pred_mask_full.astype(np.uint8), mask=pred_mask_full.astype(bool), transform=transform):
            if value == 1:
                raw_polys.append(shape(geom))

    land_geom = _load_land_mask(buffer_km=coastal_buffer_km) if (use_land_mask and georeference_status == "REAL") else None
    clean_polys, total_area_km2, total_oil_pixels = _filter_detections(
        raw_polys,
        pixel_res_deg=pixel_res_deg,
        pixel_size_m=10.0,
        min_area_km2=min_area_km2,
        land_buffered=land_geom
    )

    if clean_polys:
        merged = unary_union(clean_polys)
        slick_polygon = json.loads(json.dumps(mapping(merged)))
        area_km2 = total_area_km2
        oil_probs = prob_map[pred_mask_128 == 1]
        confidence = round(float(oil_probs.mean()), 2) if len(oil_probs) > 0 else 0.0
        estimated_age_hours = estimate_age(pred_mask_full)
    else:
        slick_polygon = None
        area_km2 = 0.0
        confidence = 0.0
        estimated_age_hours = 0.0

    return {
        "contract_version": "1.0",
        "observation_time": detection_timestamp,
        "detection_timestamp": detection_timestamp,
        "polygon": slick_polygon,
        "slick_polygon": slick_polygon,
        "area_km2": round(float(area_km2), 2),
        "age_hrs": float(estimated_age_hours),
        "estimated_age_hours": float(estimated_age_hours),
        "confidence": float(confidence),
        "georeference_status": georeference_status,
        "metadata": {
            "satellite": "Sentinel-1",
            "threshold": 0.70,
            "georeference_status": georeference_status,
        },
    }


def _process_sliding_window(
    src, model, transform, orig_h, orig_w, tile_size, image_size,
    input_scale, georeference_status, min_area_km2=0.2, use_land_mask=True,
    coastal_buffer_km=1.0
):
    """Processes a large full-swath raster using 2048x2048 sliding windows."""
    detection_timestamp = datetime.now(timezone.utc).isoformat()
    all_polygons = []
    oil_probs = []

    # Iterate in 2048x2048 steps
    for row_off in range(0, orig_h, tile_size):
        for col_off in range(0, orig_w, tile_size):
            win_h = min(tile_size, orig_h - row_off)
            win_w = min(tile_size, orig_w - col_off)
            window = Window(col_off, row_off, win_w, win_h)

            tile = src.read(1, window=window).astype(np.float32)

            # Skip tiles with >95% nodata/empty border
            valid_raw = (tile > 0) & (~np.isnan(tile))
            if valid_raw.mean() < 0.05:
                continue

            # Pad edge/partial tiles to 2048x2048
            if win_h < tile_size or win_w < tile_size:
                padded_tile = np.zeros((tile_size, tile_size), dtype=np.float32)
                padded_tile[:win_h, :win_w] = tile
                tile = padded_tile

            # Preprocess (linear to dB and normalize)
            norm, valid_mask = _preprocess_tile(tile, input_scale=input_scale)

            # Run inference at 128x128
            tile_128 = cv2.resize(norm, (image_size, image_size))
            img_tensor = torch.tensor(tile_128, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                pred = model(img_tensor)
                prob_map = torch.sigmoid(pred).squeeze().numpy()

            # Upsample back to 2048x2048 with nearest-neighbor
            pred_mask_128 = (prob_map > 0.70).astype(np.uint8)  # threshold=0.70 (sweep-optimised)
            pred_mask_tile = cv2.resize(
                pred_mask_128, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST
            )
            pred_mask_tile = pred_mask_tile & valid_mask.astype(np.uint8)

            # Crop back if padded
            if win_h < tile_size or win_w < tile_size:
                pred_mask_tile = pred_mask_tile[:win_h, :win_w]

            # Collect oil pixels and probabilities
            oil_count = int(np.count_nonzero(pred_mask_tile))
            if oil_count > 0:
                if (pred_mask_128 == 1).any():
                    oil_probs.extend(prob_map[pred_mask_128 == 1].tolist())

                # Local tile transform for polygonization
                tile_transform = rasterio.windows.transform(window, transform)
                for geom, value in shapes(
                    pred_mask_tile.astype(np.uint8),
                    mask=pred_mask_tile.astype(bool),
                    transform=tile_transform
                ):
                    if value == 1:
                        all_polygons.append(shape(geom))

    # Apply coastal land mask and minimum area filter
    land_geom = _load_land_mask(buffer_km=coastal_buffer_km) if (use_land_mask and georeference_status == "REAL") else None
    pixel_res = src.res[0] if georeference_status == "REAL" else 0.0001
    pixel_size_m = 10.0

    clean_polys, total_area_km2, total_oil_pixels = _filter_detections(
        all_polygons,
        pixel_res_deg=pixel_res,
        pixel_size_m=pixel_size_m,
        min_area_km2=min_area_km2,
        land_buffered=land_geom
    )

    if clean_polys:
        merged = unary_union(clean_polys)
        slick_polygon = json.loads(json.dumps(mapping(merged)))
        area_km2 = total_area_km2
        confidence = round(float(np.mean(oil_probs)), 2) if oil_probs else 0.0
        perimeter_pixels = merged.length / pixel_res
        compactness = perimeter_pixels / (total_oil_pixels ** 0.5) if total_oil_pixels > 0 else 0.0
        estimated_age_hours = round(min(compactness * 2, 48), 1)
    else:
        slick_polygon = None
        area_km2 = 0.0
        confidence = 0.0
        estimated_age_hours = 0.0

    return {
        "contract_version": "1.0",
        "observation_time": detection_timestamp,
        "detection_timestamp": detection_timestamp,
        "polygon": slick_polygon,
        "slick_polygon": slick_polygon,
        "area_km2": round(float(area_km2), 2),
        "age_hrs": float(estimated_age_hours),
        "estimated_age_hours": float(estimated_age_hours),
        "confidence": float(confidence),
        "georeference_status": georeference_status,
        "metadata": {
            "satellite": "Sentinel-1",
            "threshold": 0.70,
            "georeference_status": georeference_status,
        },
    }


def mask_to_geojson(mask, transform):
    """Convert a binary mask into a GeoJSON Polygon object."""
    if mask.sum() == 0:
        return None

    polygons = []
    for geom, value in shapes(mask, mask=mask.astype(bool), transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    if not polygons:
        return None

    merged = unary_union(polygons)
    raw_mapping = mapping(merged)
    return json.loads(json.dumps(raw_mapping))


def estimate_age(mask):
    """Rough age heuristic based on spread/shape irregularity."""
    area = np.count_nonzero(mask)
    if area == 0:
        return 0.0

    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask)
    perimeter = np.count_nonzero(mask) - np.count_nonzero(eroded)
    compactness = perimeter / (area ** 0.5)
    estimated_age = min(compactness * 2, 48)
    return round(estimated_age, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Oil spill detection inference.")
    default_ckpt = "unet_spill_best.pth" if os.path.exists("unet_spill_best.pth") else "unet_spill_checkpoint.pth"
    parser.add_argument("--image", type=str, default="dataset/real_dataset/images/00000.tif" if os.path.exists("dataset/real_dataset/images/00000.tif") else "dataset/backup_synthetic_200/images/00000.tif", help="Path to input Sentinel-1 SAR image (.tif)")
    parser.add_argument("--checkpoint", "--model", dest="checkpoint", type=str, default=default_ckpt, help="Path to trained .pth model checkpoint")
    parser.add_argument("--input_scale", type=str, choices=["linear", "db"], default="linear")
    parser.add_argument("--image_size", type=int, default=128, help="Model input resize dimension")
    parser.add_argument("--min_area_km2", type=float, default=0.2, help="Discard slicks smaller than this area (km2)")
    parser.add_argument("--coastal_buffer_km", type=float, default=1.0, help="Buffer distance from coastline in km to exclude")
    parser.add_argument("--no_land_mask", action="store_true", help="Disable coastline/land masking")
    parser.add_argument("--out", type=str, default="handoff_output.json", help="Path to output GeoJSON handoff file")
    args = parser.parse_args()

    # Default 00000.tif is Zenodo benchmark data in dB
    if "00000.tif" in args.image and args.input_scale == "linear":
        args.input_scale = "db"

    result = detect_spill(
        args.image,
        model_path=args.checkpoint,
        image_size=args.image_size,
        input_scale=args.input_scale,
        min_area_km2=args.min_area_km2,
        use_land_mask=not args.no_land_mask,
        coastal_buffer_km=args.coastal_buffer_km,
        save_json_path=args.out
    )
    print(f"Area: {result['area_km2']} km²")
    print(f"Estimated age: {result['estimated_age_hours']} hrs")
    print(f"Confidence: {result['confidence']}")
    print(f"Timestamp: {result['detection_timestamp']}")
    print(f"Georeference Status: {result['georeference_status']}")
    print(f"Slick Polygon: {'Generated' if result['slick_polygon'] else 'None'}")