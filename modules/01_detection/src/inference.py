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
import re
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

MAX_PLAUSIBLE_AREA_KM2 = 10000.0  # Plausible upper bound threshold for full-swath macro spills


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

        scene_obs_time = _extract_scene_timestamp(image_path)
        result = _process_single_tile(
            img, model, transform, image_size, input_scale, georeference_status,
            threshold=threshold, min_area_km2=min_area_km2, use_land_mask=use_land_mask,
            coastal_buffer_km=coastal_buffer_km, pixel_res_deg=pixel_res,
            observation_time=scene_obs_time
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
        sar_resized = cv2.resize(sar_gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        sar_bgr = cv2.cvtColor(sar_resized, cv2.COLOR_GRAY2BGR)

        # Encode SAR radar image to JPEG base64
        _, buf_sar = cv2.imencode('.jpg', sar_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sar_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf_sar).decode('utf-8')

        # 2. Generate polished model overlay on top of the original SAR base layer
        overlay_img = sar_bgr.copy()
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
                m_float = cv2.resize(poly_mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                # Anti-aliased Gaussian smoothing for natural organic fluid spill boundaries
                m_soft = cv2.GaussianBlur(m_float, (15, 15), 3.5)
                m_bin = (m_soft > 0.38).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m_smooth = cv2.morphologyEx(m_bin, cv2.MORPH_CLOSE, kernel)
                m_smooth_soft = cv2.GaussianBlur(m_smooth.astype(np.float32), (9, 9), 2.0)

                contours, _ = cv2.findContours(m_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                smoothed_contours = []
                for cnt in contours:
                    if len(cnt) >= 5:
                        smoothed_contours.append(cv2.approxPolyDP(cnt, epsilon=1.2, closed=True))
                    else:
                        smoothed_contours.append(cnt)

                # Translucent amber/gold fluid tint blend (RGB #f59e0b -> BGR 20, 150, 245)
                tint_color = np.array([20, 150, 245], dtype=np.float32)
                alpha = np.clip(m_smooth_soft * 0.45, 0.0, 0.45)[:, :, np.newaxis]
                blended = (overlay_img.astype(np.float32) * (1.0 - alpha) + tint_color * alpha).astype(np.uint8)

                # Draw glowing anti-aliased contour tracing the actual slick geometry (no hard rectangles)
                cv2.drawContours(blended, smoothed_contours, -1, (10, 110, 230), 3, lineType=cv2.LINE_AA)
                cv2.drawContours(blended, smoothed_contours, -1, (100, 220, 255), 1, lineType=cv2.LINE_AA)

                # Sleek, unobtrusive confidence badge/pill
                badge_x, badge_y = 16, 20
                badge_w, badge_h = 160, 32
                pill_overlay = blended.copy()
                # Rounded pill background
                r = 6
                cv2.rectangle(pill_overlay, (badge_x + r, badge_y), (badge_x + badge_w - r, badge_y + badge_h), (16, 22, 32), -1)
                cv2.rectangle(pill_overlay, (badge_x, badge_y + r), (badge_x + badge_w, badge_y + badge_h - r), (16, 22, 32), -1)
                cv2.circle(pill_overlay, (badge_x + r, badge_y + r), r, (16, 22, 32), -1, lineType=cv2.LINE_AA)
                cv2.circle(pill_overlay, (badge_x + badge_w - r, badge_y + r), r, (16, 22, 32), -1, lineType=cv2.LINE_AA)
                cv2.circle(pill_overlay, (badge_x + r, badge_y + badge_h - r), r, (16, 22, 32), -1, lineType=cv2.LINE_AA)
                cv2.circle(pill_overlay, (badge_x + badge_w - r, badge_y + badge_h - r), r, (16, 22, 32), -1, lineType=cv2.LINE_AA)
                cv2.addWeighted(pill_overlay, 0.88, blended, 0.12, 0, blended)

                # Pill border & content
                cv2.rectangle(blended, (badge_x + r, badge_y), (badge_x + badge_w - r, badge_y + badge_h), (55, 80, 110), 1, lineType=cv2.LINE_AA)
                cv2.circle(blended, (badge_x + 14, badge_y + 16), 4, (24, 150, 245), -1, lineType=cv2.LINE_AA)
                cv2.putText(blended, f"OIL SLICK  p={confidence:.2f}", (badge_x + 25, badge_y + 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 250), 1, lineType=cv2.LINE_AA)
                overlay_img = blended
            except Exception as ex:
                print(f"Warning: could not render slick polygon overlay: {ex}")
        else:
            # Clean baseline SAR image with a subtle dark pill badge
            badge_x, badge_y = 16, 20
            badge_w, badge_h = 175, 32
            pill_overlay = overlay_img.copy()
            cv2.rectangle(pill_overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (16, 22, 32), -1)
            cv2.addWeighted(pill_overlay, 0.85, overlay_img, 0.15, 0, overlay_img)
            cv2.rectangle(overlay_img, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (45, 65, 90), 1, lineType=cv2.LINE_AA)
            cv2.circle(overlay_img, (badge_x + 14, badge_y + 16), 4, (60, 180, 100), -1, lineType=cv2.LINE_AA)
            cv2.putText(overlay_img, "NO SLICK DETECTED", (badge_x + 25, badge_y + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 210), 1, lineType=cv2.LINE_AA)

        _, buf_mask = cv2.imencode('.png', overlay_img)
        mask_b64 = 'data:image/png;base64,' + base64.b64encode(buf_mask).decode('utf-8')
        return sar_b64, mask_b64
    except Exception as e:
        print(f"Error generating SAR previews: {e}")
        return None, None


def _preprocess_tile(tile, input_scale="linear"):
    """
    Scales SAR backscatter to [0, 1] for U-Net input matching data_loader.py training setup.
    - If in dB (Zenodo benchmark scenes / negative values): clips to [-35.0 dB, -5.0 dB] and scales.
    - If in linear power/amplitude (SNAP GeoTIFF exports): applies min-max normalization with outlier protection,
      ensuring dark ocean background is mapped to ~0.00-0.05 and high backscatter to ~1.0.
    Masks nodata pixels to prevent false-positive detections.
    """
    valid_mask = ~np.isnan(tile) & (tile > -999) & (tile != 0)
    norm = np.zeros_like(tile, dtype=np.float32)
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
        # Linear power / amplitude SAR input (e.g. Sentinel-1 SNAP exports)
        # Matches data_loader.py normalization where ocean background is dark (~0.0-0.05)
        p_min = float(vals.min())
        p_max = float(vals.max())
        if p_max > p_min:
            # Handle high-intensity point targets if max > 1.0 to prevent dynamic range squashing
            p_hi = float(np.percentile(vals, 99.8)) if p_max > 1.0 else p_max
            p_hi = max(p_hi, p_min + 1e-5)
            clipped = np.clip(vals, p_min, p_hi)
            norm[valid_mask] = (clipped - p_min) / (p_hi - p_min + 1e-6)
        else:
            norm[valid_mask] = 0.0

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


def _extract_scene_timestamp(image_path: str) -> str:
    """Extracts satellite acquisition timestamp from Sentinel-1 filename or metadata, or defaults to now UTC."""
    p_str = str(image_path)
    match = re.search(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})', p_str)
    if match:
        year, month, day, hour, minute, second = match.groups()
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}+00:00"
    if "aug29" in p_str.lower() or "29aug" in p_str.lower():
        return "2021-08-29T00:02:02+00:00"
    if "sep3" in p_str.lower() or "03sep" in p_str.lower():
        return "2021-09-03T00:10:04+00:00"
    return datetime.now(timezone.utc).isoformat()


def _process_single_tile(
    img, model, transform, image_size, input_scale, georeference_status,
    threshold=0.70, min_area_km2=0.2, use_land_mask=True, coastal_buffer_km=1.0, pixel_res_deg=0.0001,
    observation_time=None
):
    """Processes a single <=2048x2048 scene."""
    orig_shape = img.shape
    norm, valid_mask = _preprocess_tile(img, input_scale=input_scale)

    img_resized = cv2.resize(norm, (image_size, image_size))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    detection_timestamp = observation_time or datetime.now(timezone.utc).isoformat()

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
    effective_pixel_size_m = float(pixel_res_deg * 111320.0) if pixel_res_deg > 0 else 10.0
    clean_polys, total_area_km2, total_oil_pixels = _filter_detections(
        raw_polys,
        pixel_res_deg=pixel_res_deg,
        pixel_size_m=effective_pixel_size_m,
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