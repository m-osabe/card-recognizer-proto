"""
test_recovery.py
新検出器による card-10 失敗写真の救済数テスト
"""
import cv2
import numpy as np
import json
from card_detector import four_point_transform
from card_engine import CardRecognitionEngine

def detect_card_quad_improved(binary_img, min_area, max_area):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            break
        if area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(approx)
                wb, hb = rect[1]
                if wb > 0 and hb > 0 and 1.15 <= max(wb, hb)/min(wb, hb) <= 2.10:
                    return approx
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < min_area or hull_area > max_area:
            continue
        h_peri = cv2.arcLength(hull, True)
        for eps in [0.025, 0.035, 0.045, 0.055]:
            approx_h = cv2.approxPolyDP(hull, eps * h_peri, True)
            if len(approx_h) == 4 and cv2.isContourConvex(approx_h):
                rect = cv2.minAreaRect(approx_h)
                wb, hb = rect[1]
                if wb > 0 and hb > 0 and 1.15 <= max(wb, hb)/min(wb, hb) <= 2.10:
                    return approx_h
        rect = cv2.minAreaRect(hull)
        wb, hb = rect[1]
        if wb > 0 and hb > 0:
            asp = max(wb, hb) / min(wb, hb)
            if 1.18 <= asp <= 2.0 and (hull_area / (wb*hb)) > 0.72:
                box = cv2.boxPoints(rect).astype(np.int32)
                return box.reshape(4, 1, 2)
    return None

engine = CardRecognitionEngine()
engine.load_index()

with open('output/failure_analysis.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for target_id in ['card-6', 'card-10', 'card-13']:
    target_fails = [f['file_name'] for f in d['failures'] if f['ground_truth'] == target_id]
    recovered = 0
    print(f"\n=== Testing recovery for {target_id} ({len(target_fails)} failures) ===", flush=True)
    for fn in target_fails:
        img = cv2.imread('data/test/' + fn)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = 800.0 / max(h, w)
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 30, 120)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        q = detect_card_quad_improved(edges, (h*w*scale**2)*0.05, (h*w*scale**2)*0.95)
        if q is None:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            q = detect_card_quad_improved(thresh, (h*w*scale**2)*0.05, (h*w*scale**2)*0.95)
        if q is None:
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            q = detect_card_quad_improved(otsu, (h*w*scale**2)*0.05, (h*w*scale**2)*0.95)
            
        if q is not None:
            corners = (q.reshape(4, 2) / scale).astype('float32')
            crop = four_point_transform(img, corners, (600, 840))
            res = engine.identify(crop, top_k=1)
            best = res.get('best_match', {})
            cid = best.get('card_id')
            if cid == target_id:
                recovered += 1
                sc = best.get('combined_score', 0)
                inl = best.get('inliers', 0)
                print(f"  [RECOVERED] {fn} -> {cid} (Score: {sc:.1f}, Inl: {inl})", flush=True)

    print(f"--> {target_id}: {recovered} / {len(target_fails)} recovered!", flush=True)
