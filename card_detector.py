"""
card_detector.py
カード写真からカードの四隅（頂点）を検出し、正面の長方形に射影変換（透視変換）するモジュール
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    4つの頂点座標を [左上, 右上, 右下, 左下] の順に並び替える
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 左上は x+y が最小、右下は x+y が最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上は y-x (または x-y) の差分で判定
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(
    image: np.ndarray, pts: np.ndarray, output_size: Tuple[int, int] = (600, 840)
) -> np.ndarray:
    """
    4点座標に基づいて透視変換を行い、正面の長方形画像を切り出す
    output_size: (width, height)
    """
    rect = order_points(pts)
    dst_w, dst_h = output_size

    dst = np.array(
        [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
        dtype="float32",
    )

    # 変換行列を計算して適用
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (dst_w, dst_h))

    return warped


class CardDetector:
    def __init__(
        self,
        target_width: int = 600,
        target_height: int = 840,
        min_card_area_ratio: float = 0.05,
    ):
        """
        target_width, target_height: 切り出し後のカード解像度（標準縦横比 約 1:1.4）
        min_card_area_ratio: 写真全体に対するカードの最小面積比率
        """
        self.target_width = target_width
        self.target_height = target_height
        self.min_card_area_ratio = min_card_area_ratio

    def detect_and_crop(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        """
        写真からカードを検出して正面画像を切り出す。
        Returns:
            cropped_card: 正面に補正されたカード画像（検出失敗時はリサイズした元画像）
            corners: 検出された4頂点（検出失敗時は None）
            meta: 検出に関するメタ情報
        """
        orig_h, orig_w = image.shape[:2]
        total_area = orig_h * orig_w
        min_area = total_area * self.min_card_area_ratio

        # 処理高速化・ノイズ低減のために縮小して輪郭探索
        scale = 800.0 / max(orig_h, orig_w) if max(orig_h, orig_w) > 800 else 1.0
        if scale < 1.0:
            small_img = cv2.resize(
                image, (int(orig_w * scale), int(orig_h * scale))
            )
        else:
            small_img = image.copy()

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        card_contour = None

        # 1. Canny法 (エッジ強調)
        edges = cv2.Canny(blurred, 30, 120)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        card_contour = self._find_quad_contour(edges, min_area * (scale**2), total_area * (scale**2) * 0.95)

        # 2. 失敗した場合は適応的二値化で再試行
        if card_contour is None:
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
            card_contour = self._find_quad_contour(thresh, min_area * (scale**2), total_area * (scale**2) * 0.95)

        # 3. それでも見つからない場合はOtsu二値化
        if card_contour is None:
            _, otsu = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            card_contour = self._find_quad_contour(otsu, min_area * (scale**2), total_area * (scale**2) * 0.95)

        if card_contour is not None:
            # 元画像のスケールに戻す
            corners = (card_contour.reshape(4, 2) / scale).astype("float32")

            # 4点透視変換を実行
            cropped = four_point_transform(
                image, corners, (self.target_width, self.target_height)
            )
            return cropped, corners, {"detected": True, "corners": corners.tolist()}
        else:
            # 輪郭が見つからなかった場合は、中央トリミングまたは元画像リサイズ
            cropped = cv2.resize(image, (self.target_width, self.target_height))
            return cropped, None, {"detected": False, "reason": "No card quad found"}

    def _find_quad_contour(
        self, binary_img: np.ndarray, min_area: float, max_area: float = 1e9
    ) -> Optional[np.ndarray]:
        """二値画像から ConvexHull 凸包・多段階近似・回転矩形フィッティングで4頂点輪郭を検出"""
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                break
            if area > max_area:
                continue

            # Stage 1: 生輪郭の直接近似 (正方形〜長方形)
            peri = cv2.arcLength(cnt, True)
            for eps in [0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    rect = cv2.minAreaRect(approx)
                    wb, hb = rect[1]
                    if wb > 0 and hb > 0:
                        asp = max(wb, hb) / min(wb, hb)
                        if 1.15 <= asp <= 2.10:
                            return approx

            # Stage 2: ConvexHull (凸包) で手持ち指のかぶり・角丸の凹凸を修復
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
                    if wb > 0 and hb > 0:
                        asp = max(wb, hb) / min(wb, hb)
                        if 1.15 <= asp <= 2.10:
                            return approx_h

            # Stage 3: minAreaRect 回転矩形フィッティング (指で輪郭が大きく削れている場合)
            rect = cv2.minAreaRect(hull)
            wb, hb = rect[1]
            if wb > 0 and hb > 0:
                asp = max(wb, hb) / min(wb, hb)
                rect_area = wb * hb
                if 1.18 <= asp <= 2.05 and (hull_area / rect_area) > 0.72:
                    box = cv2.boxPoints(rect).astype(np.int32)
                    return box.reshape(4, 1, 2)

        return None
