"""
visualize_detection_debug.py
カード検出処理（エッジ、適応的二値化、Otsu、輪郭抽出、SIFTインライア）の各内部状態を画像として保存し、
なぜ枠検出が成功/失敗したかを視覚的に分析・デバッグするためのスクリプト
"""

import os
import cv2
import numpy as np
import argparse
from card_detector import CardDetector
from matcher_sift import SIFTCardMatcher


def debug_visualize(image_path: str, output_dir: str = "output/debug_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"画像を読み込めませんでした: {image_path}")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    orig_h, orig_w = img.shape[:2]
    scale = 800.0 / max(orig_h, orig_w) if max(orig_h, orig_w) > 800 else 1.0
    small_img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))

    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Canny
    edges = cv2.Canny(blurred, 50, 150)
    dilated_edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_canny.jpg"), edges)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_canny_dilated.jpg"), dilated_edges)

    # 2. Adaptive
    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_adaptive.jpg"), adapt)

    # 3. Otsu
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_otsu.jpg"), otsu)

    # 4. 輪郭の描画 (RETR_EXTERNAL)
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    vis_cnt = small_img.copy()
    for i, cnt in enumerate(contours[:5]):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        is_convex = cv2.isContourConvex(approx)
        color = (0, 255, 0) if len(approx) == 4 and is_convex else (0, 0, 255)
        cv2.drawContours(vis_cnt, [cnt], -1, (255, 255, 0), 1)
        cv2.drawContours(vis_cnt, [approx], -1, color, 2)
        for pt in approx:
            cv2.circle(vis_cnt, tuple(pt[0]), 4, (255, 0, 0), -1)

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_contours.jpg"), vis_cnt)
    print(f"デバッグ画像を保存しました: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug detection visualization")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="output/debug_analysis", help="Output directory")
    args = parser.parse_args()
    debug_visualize(args.image, args.output_dir)
