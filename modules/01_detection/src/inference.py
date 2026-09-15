# inference.py
import os
import sys
from pathlib import Path
_module_dir = Path(__file__).resolve().parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

import argparse
import base64
import json
from datetime import datetime, timezone
import cv2
import numpy as np
import rasterio
import rasterio.enums
from rasterio.features import shapes, rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import torch

from model import UNet, load_spill_model
from validation import is_valid_sar_image

MAX_PLAUSIBLE_AREA_KM2 = 500.0  # Sanity check threshold for single-spill detections


def detect_spill(
    image_path,
    model_path="unet_spill_best.pth",
    image_size=128,
    tile_size=2048,
    input_scale="linear",
    threshold=0.70,
    min_area_km2=0.2,
    use_land_mask=True,
    coastal_buffer_km=1.0,
    save_json_path=None,
    preloaded_model=None
):
    """
    Run oil spill detection on a GeoTIFF scene.
    Supports both localized patches (<=2048x2048) and full Sentinel-1 GRD swaths (via sliding window).
    Includes post-processing filters:
      - Coastal/Land mask with buffer to exclude false positives along shores and wetlands
      - Minimum area threshold to suppress micro-speckle false alarms
    """
    # 1. Validation check FIRST before model loading or preprocessing
    is_valid, validation_msg = is_valid_sar_image(image_path)
    if not is_valid:
        invalid_response = {
            "status": "invalid_input",
            "message": validation_msg or (
                "Uploaded file does not appear to be a valid SAR satellite image. "
                "Expected Sentinel-1 characteristics (1-2 band radar backscatter data) were not found."
            ),
            "slick_polygon": None,
            "polygon": None,
            "area_km2": None,
        }
        if save_json_path:
            with open(save_json_path, "w") as f:
                json.dump(invalid_response, f, indent=2)
            print(f"Saved invalid input report to {save_json_path}")
        return invalid_response

    if preloaded_model is not None:
        model = preloaded_model
    else:
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

        # Fast decimation on read for swaths > 2048 to enable lightning-fast inference in seconds
        import gc
        torch.set_num_threads(1)
        max_proc_dim = 2048
        if orig_h > max_proc_dim or orig_w > max_proc_dim:
            decimate = max(1, orig_h // max_proc_dim, orig_w // max_proc_dim)
            out_shape = (orig_h // decimate, orig_w // decimate)
            img = src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
            transform = transform * transform.scale(
                (orig_w / out_shape[1]),
                (orig_h / out_shape[0])
            )
            pixel_res = (src.res[0] * decimate) if has_real_crs else (0.0001 * decimate)
        else:
            img = src.read(1).astype(np.float32)
            pixel_res = src.res[0] if has_real_crs else 0.0001

        result = _process_single_tile(
            img, model, transform, image_size, input_scale, georeference_status,
            threshold=threshold, min_area_km2=min_area_km2, use_land_mask=use_land_mask,
            coastal_buffer_km=coastal_buffer_km, pixel_res_deg=pixel_res
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

    # Derive regional label from slick polygon centroid or bounds
    if result.get("slick_polygon"):
        try:
            poly_shape = shape(result["slick_polygon"])
            c_lon, c_lat = poly_shape.centroid.x, poly_shape.centroid.y
            if 50 <= c_lat <= 62 and -4 <= c_lon <= 12:
                region_name = "North Sea · Offshore Sector"
            elif 24 <= c_lat <= 32 and -98 <= c_lon <= -80:
                region_name = "Gulf of Mexico · Deepwater Sector"
            elif 15 <= c_lat <= 24 and 68 <= c_lon <= 76:
                region_name = "Arabian Sea · Mumbai Offshore Sector"
            else:
                lat_str = f"{abs(c_lat):.2f}°{'N' if c_lat >= 0 else 'S'}"
                lon_str = f"{abs(c_lon):.2f}°{'E' if c_lon >= 0 else 'W'}"
                region_name = f"Marine Sector ({lat_str}, {lon_str})"
            result["metadata"]["region"] = region_name
            result["region"] = region_name
            result["centroid"] = [c_lon, c_lat]
        except Exception:
            pass

    # Generate visual preview images from the real SAR scene and detected mask
    try:
        sar_b64, mask_b64 = _generate_sar_previews(
            image_path=image_path,
            slick_polygon=result.get("slick_polygon"),
            confidence=result.get("confidence", 0.0),
            has_detection=(result.get("area_km2", 0.0) > 0 and result.get("slick_polygon") is not None),
            preloaded_tile=img,
            preloaded_transform=transform
        )
        if sar_b64 and mask_b64:
            result["preview"] = {
                "radar_image": sar_b64,
                "mask_image": mask_b64
            }
    except Exception as e:
        print(f"Warning: could not generate preview images: {e}")

    try:
        del img
        gc.collect()
    except Exception:
        pass

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


def _generate_sar_previews(
    image_path,
    slick_polygon,
    confidence=0.0,
    has_detection=False,
    target_w=440,
    target_h=760,
    preloaded_tile=None,
    preloaded_transform=None
):
    """
    Generates real base64-encoded visual previews of the SAR radar scene and detected oil slick mask
    for direct native display in the frontend HTML5 canvas/image elements.
    """
    try:
        if preloaded_tile is not None:
            tile = preloaded_tile
            transform = preloaded_transform
            out_shape = tile.shape
        else:
            with rasterio.open(image_path) as src:
                orig_h, orig_w = src.height, src.width
                decimate = max(1, orig_h // target_h, orig_w // target_w)
                out_shape = (max(1, orig_h // decimate), max(1, orig_w // decimate))
                tile = src.read(1, out_shape=out_shape, resampling=rasterio.enums.Resampling.nearest).astype(np.float32)
                transform = src.transform * src.transform.scale(
                    (orig_w / out_shape[1]),
                    (orig_h / out_shape[0])
                )

        # 1. Normalize SAR backscatter for display
        valid = ~np.isnan(tile) & (tile > -999) & (tile != 0)
        if np.any(valid):
            p2, p98 = np.percentile(tile[valid], (2, 98))
            norm = np.clip((tile - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)
        else:
            norm = np.zeros_like(tile)

        sar_gray = (norm * 255).astype(np.uint8)
        sar_resized = cv2.resize(sar_gray, (target_w, target_h))

        # Encode SAR radar image to JPEG base64
        _, buf_sar = cv2.imencode('.jpg', sar_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sar_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf_sar).decode('utf-8')

        # 2. Generate model output mask visualization
        mask_bg = np.full((target_h, target_w, 3), (24, 20, 14), dtype=np.uint8)
        if has_detection and slick_polygon:
            geom = shape(slick_polygon)
            try:
                poly_mask = rasterize(
                    [(geom, 1)],
                    out_shape=out_shape,
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                )
                m_resized = cv2.resize(poly_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                m_resized = np.zeros((target_h, target_w), dtype=np.uint8)

            if np.any(m_resized == 1):
                # Amber/gold slick fill (#d97706 -> BGR: 6, 119, 217)
                mask_bg[m_resized == 1] = [30, 150, 240]
                ys, xs = np.where(m_resized == 1)
                x_min, x_max = max(0, int(xs.min()) - 8), min(target_w - 1, int(xs.max()) + 8)
                y_min, y_max = max(0, int(ys.min()) - 8), min(target_h - 1, int(ys.max()) + 8)
                cv2.rectangle(mask_bg, (x_min, y_min), (x_max, y_max), (120, 200, 40), 2)
                cv2.putText(mask_bg, f'oil p={confidence:.2f}', (x_min, max(24, y_min - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 50), 2)
            else:
                cv2.putText(mask_bg, f'oil p={confidence:.2f}', (int(target_w * 0.1), int(target_h * 0.2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 50), 2)
        else:
            cv2.rectangle(mask_bg, (int(target_w * 0.1), int(target_h * 0.15)), (int(target_w * 0.9), int(target_h * 0.85)), (60, 140, 60), 2)
            cv2.putText(mask_bg, 'No slick detected', (int(target_w * 0.2), int(target_h * 0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 200, 120), 2)

        _, buf_mask = cv2.imencode('.png', mask_bg)
        mask_b64 = 'data:image/png;base64,' + base64.b64encode(buf_mask).decode('utf-8')
        return sar_b64, mask_b64
    except Exception as e:
        print(f"Error generating SAR previews: {e}")
        return None, None


def _preprocess_tile(tile, input_scale="linear"):
    """
    Applies linear-to-dB conversion (if linear) and scales to [0, 1] for U-Net input.
    Masks nodata pixels to prevent false-positive detections.
    Auto-detects dB vs linear scale seamlessly from data range.
    """
    valid_mask = ~np.isnan(tile) & (tile > -999) & (tile != 0)
    norm = np.full_like(tile, 0.5, dtype=np.float32)  # Neutral ocean background for nodata
    if not np.any(valid_mask):
        return norm, valid_mask

    vals = tile[valid_mask]
    is_db = (input_scale == "db") or np.any(vals < 0)

    if is_db:
        # Data already in dB (Zenodo benchmark scenes)
        vmin, vmax = -35.0, -5.0
        clipped = np.clip(vals, vmin, vmax)
        norm[valid_mask] = (clipped - vmin) / (vmax - vmin)
    else:
        # Input is in linear power/amplitude (e.g. SNAP GeoTIFF exports)
        db = 10.0 * np.log10(np.maximum(vals, 1e-6))
        vmin, vmax = -35.0, -5.0
        norm[valid_mask] = np.clip((db - vmin) / (vmax - vmin), 0.0, 1.0)

    return norm, valid_mask


_CACHED_LAND_GEOM = None

def _load_land_mask(land_geojson_path="gulf_coast_land.geojson", buffer_km=1.0):
    """
    Loads land geometry and applies a coastal buffer outward into the ocean (in kilometers).
    Uses local GeoJSON if available.
    """
    global _CACHED_LAND_GEOM
    if _CACHED_LAND_GEOM is not None:
        return _CACHED_LAND_GEOM

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

    if land_geom is not None and buffer_km > 0:
        deg_buf = buffer_km / 105.0
        land_geom = land_geom.buffer(deg_buf)

    _CACHED_LAND_GEOM = land_geom
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
    threshold=0.70, min_area_km2=0.2, use_land_mask=True, coastal_buffer_km=1.0, pixel_res_deg=0.0001
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
    pred_mask_128 = (prob_map > threshold).astype(np.uint8)
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
            "threshold": threshold,
            "georeference_status": georeference_status,
        },
    }


def _process_sliding_window(
    src, model, transform, orig_h, orig_w, tile_size, image_size,
    input_scale, georeference_status, threshold=0.70, min_area_km2=0.2, use_land_mask=True,
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
            pred_mask_128 = (prob_map > threshold).astype(np.uint8)
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
            "threshold": threshold,
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
    parser.add_argument("--threshold", type=float, default=0.70, help="Oil spill detection threshold (default: 0.70)")
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
        threshold=args.threshold,
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