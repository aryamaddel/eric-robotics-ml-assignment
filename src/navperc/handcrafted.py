"""Hand-crafted (colour + shape) detectors and a stop-sign verifier.

Classical cues for the two classes that have a distinctive appearance:

  stop sign  red region (two hue bands, high saturation) whose outer contour is a convex,
             roughly square, many-sided polygon; inside it, white pixels form *text*, i.e. a
             band of mixed white/red, rather than the single solid white bar of a No Entry sign.
  cone       orange region; a cone's orange is broken up by white reflective bands, so blobs
             are merged with a vertical closing first; then the merged region must be taller
             than wide and taper towards the top.

Barriers (concrete / metal / water-filled, many colours) have no reliable colour or shape
signature, so no hand-crafted barrier detector is attempted.

Besides standalone detection, `stop_sign_features` can verify the CNN's stop-sign detections:
the No Entry confusion found in the audit is exactly "red, round-ish sign with white inside", and
the white-bar test separates the two.
"""

import cv2
import numpy as np

# deliberately loose: road-camera colours are often washed out (salmon-pink stop signs)
RED_LO1, RED_HI1 = (0, 45, 40), (12, 255, 255)
RED_LO2, RED_HI2 = (155, 45, 40), (180, 255, 255)
ORANGE_LO, ORANGE_HI = (3, 80, 70), (22, 255, 255)  # loose: washed-out road-camera colours
WHITE_LO, WHITE_HI = (0, 0, 150), (180, 70, 255)


def red_mask(hsv):
    return cv2.inRange(hsv, RED_LO1, RED_HI1) | cv2.inRange(hsv, RED_LO2, RED_HI2)


def stop_sign_features(crop_bgr) -> dict:
    """Shape/colour features of a sign crop (tight box around the candidate)."""
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return {"valid": False}
    crop = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = red_mask(hsv)
    # adaptive white: "whiteness" = brightness - saturation, split per crop with Otsu
    whiteness = np.clip(hsv[..., 2].astype(np.int16) - hsv[..., 1].astype(np.int16), 0, 255).astype(np.uint8)
    _, white = cv2.threshold(whiteness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"valid": False, "red_frac": 0.0}
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    solidity = area / (cv2.contourArea(hull) + 1e-6)
    peri = cv2.arcLength(hull, True)
    vertices = len(cv2.approxPolyDP(hull, 0.03 * peri, True))
    sign = np.zeros_like(red)
    cv2.drawContours(sign, [hull], -1, 255, -1)  # filled sign area incl. white symbol
    inside = sign > 0
    # centre band of the sign (rows 38-62%, cols 22-78%): solid white for No Entry, letters for STOP
    band = np.zeros_like(inside)
    band[24:40, 14:50] = True
    band &= inside
    band_white = float((white[band] > 0).mean()) if band.any() else 0.0
    # rows of the band that are almost entirely white = a bar
    rows = [(white[r, 14:50] > 0).mean() for r in range(24, 40)]
    bar_rows = float(np.mean([v > 0.85 for v in rows]))
    return {
        "valid": True,
        "red_frac": float((red > 0).mean()),
        "fill": float(inside.mean()),
        "solidity": float(solidity),
        "vertices": int(vertices),
        "band_white": band_white,
        "bar_rows": bar_rows,
    }


def looks_like_no_entry(f: dict) -> bool:
    """A solid white horizontal bar across the middle of a red sign."""
    return f.get("valid", False) and f["bar_rows"] >= 0.53  # tuned on the v1 audit crops


def verify_stop_sign(crop_bgr) -> bool:
    """True unless the crop is clearly not a stop sign (no red, or a No Entry style bar)."""
    f = stop_sign_features(crop_bgr)
    if not f.get("valid") or f["red_frac"] < 0.15:
        return False
    return not looks_like_no_entry(f)


def detect_stop_signs(img_bgr, min_side=12):
    """Standalone hand-crafted stop-sign detector -> list of (score, (x0, y0, x1, y1))."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.morphologyEx(red_mask(hsv), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = []
    for c in cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if min(w, h) < min_side or not 0.75 < w / h < 1.33:
            continue
        hull = cv2.convexHull(c)
        solidity = cv2.contourArea(c) / (cv2.contourArea(hull) + 1e-6)
        peri = cv2.arcLength(hull, True)
        circularity = 4 * np.pi * cv2.contourArea(hull) / (peri ** 2 + 1e-6)
        vertices = len(cv2.approxPolyDP(hull, 0.03 * peri, True))
        if solidity < 0.75 or not 6 <= vertices <= 10 or circularity < 0.75:
            continue
        f = stop_sign_features(img_bgr[y:y + h, x:x + w])
        if not f.get("valid") or looks_like_no_entry(f) or f["band_white"] < 0.1:
            continue  # stop signs carry white text; plain red discs/octagons without it are rejected
        score = circularity * solidity * min(1.0, f["band_white"] / 0.3)
        out.append((float(score), (x, y, x + w, y + h)))
    return out


def detect_cones(img_bgr, min_h=12):
    """Standalone hand-crafted cone detector -> list of (score, (x0, y0, x1, y1))."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    # bridge the white reflective bands: close vertically, then a little horizontally
    merged = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15)))
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = []
    for c in cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if h < min_h or not 1.0 <= h / w <= 6.0:
            continue
        blob = merged[y:y + h, x:x + w] > 0
        fill = blob.mean()
        if fill < 0.35:
            continue
        # nuImages "traffic cones" are mostly straight-sided channeliser posts, so tapering is a
        # bonus rather than a requirement
        q = max(1, h // 4)
        taper = 1 - blob[:q].sum(1).mean() / (blob[-q:].sum(1).mean() + 1e-6)
        orange_frac = (orange[y:y + h, x:x + w] > 0).mean()
        score = float(min(1.0, fill / 0.6) * min(1.0, orange_frac / 0.4) * (0.7 + 0.3 * min(1.0, max(0.0, taper) / 0.6)))
        out.append((score, (x, y, x + w, y + h)))
    return out
