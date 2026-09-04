"""
card_detector.py
カード写真からカードの四隅（頂点）を検出し、正面の長方形に射影変換（透視変換）するモジュール
"""

import cv2
import numpy as np
from itertools import combinations
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
        max_card_area_ratio: float = 0.55,
    ):
        """
        target_width, target_height: 切り出し後のカード解像度（標準縦横比 約 1:1.4）
        min_card_area_ratio: 写真全体に対するカードの最小面積比率 (5%)
        max_card_area_ratio: 写真全体に対するカードの最大面積比率 (55%: 机全体の影など巨大台形を排除)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.min_card_area_ratio = min_card_area_ratio
        self.max_card_area_ratio = max_card_area_ratio

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
        max_area = total_area * self.max_card_area_ratio

        # 処理高速化・ノイズ低減のために縮小して輪郭探索
        scale = 800.0 / max(orig_h, orig_w) if max(orig_h, orig_w) > 800 else 1.0
        if scale < 1.0:
            small_img = cv2.resize(
                image, (int(orig_w * scale), int(orig_h * scale))
            )
        else:
            small_img = image.copy()

        sh, sw = small_img.shape[:2]
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        card_contour = None

        # 1. Canny法 (適正閾値 40, 140 で机の影グラデーションを遮断)
        edges = cv2.Canny(blurred, 40, 140)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        card_contour = self._find_quad_contour(edges, min_area * (scale**2), max_area * (scale**2), sw, sh)

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
            card_contour = self._find_quad_contour(thresh, min_area * (scale**2), max_area * (scale**2), sw, sh)

        # 3. それでも見つからない場合はOtsu二値化
        if card_contour is None:
            _, otsu = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            card_contour = self._find_quad_contour(otsu, min_area * (scale**2), max_area * (scale**2), sw, sh)

        # 4. それでも見つからない場合は複合輪郭から長方形サブセット探索 (手持ち腕・指の合体輪郭からカード四隅を救済)
        if card_contour is None:
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for cnt in cnts[:3]:
                if cv2.contourArea(cnt) >= min_area * (scale**2) * 0.5:
                    sub_quad = self._find_sub_quadrilateral(cnt, min_area * (scale**2), max_area * (scale**2), sw, sh)
                    if sub_quad is not None:
                        card_contour = sub_quad
                        break

        if card_contour is not None:
            # 元画像のスケールに戻す
            corners = (card_contour.reshape(4, 2) / scale).astype("float32")

            # 4点透視変換を実行
            cropped = four_point_transform(
                image, corners, (self.target_width, self.target_height)
            )
            return cropped, corners, {"detected": True, "corners": corners.tolist(), "detection_method": "bottom_up_contour"}
        else:
            # 輪郭が見つからなかった場合は、中央トリミングまたは元画像リサイズ
            cropped = cv2.resize(image, (self.target_width, self.target_height))
            return cropped, None, {"detected": False, "reason": "No card quad found"}

    def _find_quad_contour(
        self, binary_img: np.ndarray, min_area: float, max_area: float, img_w: int, img_h: int
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
            for eps in [0.02, 0.03]:
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if not self._is_touching_image_border(approx, img_w, img_h):
                        rect = cv2.minAreaRect(approx)
                        wb, hb = rect[1]
                        if wb > 0 and hb > 0 and 1.15 <= max(wb, hb) / min(wb, hb) <= 2.10:
                            return approx

            # Stage 2: ConvexHull (凸包) で手持ち指のかぶり・角丸の凹凸を修復
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area < min_area or hull_area > max_area:
                continue

            h_peri = cv2.arcLength(hull, True)
            for eps in [0.025, 0.035, 0.045]:
                approx_h = cv2.approxPolyDP(hull, eps * h_peri, True)
                if len(approx_h) == 4 and cv2.isContourConvex(approx_h):
                    if not self._is_touching_image_border(approx_h, img_w, img_h):
                        rect = cv2.minAreaRect(approx_h)
                        wb, hb = rect[1]
                        if wb > 0 and hb > 0 and 1.15 <= max(wb, hb) / min(wb, hb) <= 2.10:
                            return approx_h

            # Stage 3: minAreaRect 回転矩形フィッティング (指で輪郭が大きく削れている場合)
            rect = cv2.minAreaRect(hull)
            wb, hb = rect[1]
            if wb > 0 and hb > 0:
                asp = max(wb, hb) / min(wb, hb)
                rect_area = wb * hb
                if 1.18 <= asp <= 2.05 and (hull_area / rect_area) > 0.80:
                    box = cv2.boxPoints(rect).astype(np.int32).reshape(4, 1, 2)
                    if not self._is_touching_image_border(box, img_w, img_h):
                        return box

        return None

    def _find_sub_quadrilateral(
        self, cnt: np.ndarray, min_area: float, max_area: float, img_w: int, img_h: int
    ) -> Optional[np.ndarray]:
        """複合輪郭（腕・指などが合体した輪郭）の多角形頂点群からカード長方形を探索"""
        peri = cv2.arcLength(cnt, True)
        for eps_ratio in [0.012, 0.018]:
            approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
            pts = approx.reshape(-1, 2)
            if len(pts) < 4 or len(pts) > 25:
                continue

            best_score = -1e9
            best_quad = None

            for quad_idx in combinations(range(len(pts)), 4):
                q = pts[list(quad_idx)].astype("float32")
                if not cv2.isContourConvex(q.astype(np.int32)):
                    continue
                ordered = order_points(q)
                w1 = np.linalg.norm(ordered[0] - ordered[1])
                w2 = np.linalg.norm(ordered[3] - ordered[2])
                h1 = np.linalg.norm(ordered[0] - ordered[3])
                h2 = np.linalg.norm(ordered[1] - ordered[2])
                w = (w1 + w2) / 2.0
                h = (h1 + h2) / 2.0
                if w < 40 or h < 40:
                    continue
                asp = max(w, h) / (min(w, h) + 1e-5)
                if not (1.20 <= asp <= 1.65):
                    continue
                area = cv2.contourArea(ordered.astype(np.int32))
                if area < min_area * 0.5 or area > max_area:
                    continue

                if self._is_touching_image_border(ordered, img_w, img_h):
                    continue

                # 4つの内角の直交性チェック (90度との平均誤差)
                def angle(p1, p2, p3):
                    v1 = p1 - p2
                    v2 = p3 - p2
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-5)
                    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

                a1 = angle(ordered[3], ordered[0], ordered[1])
                a2 = angle(ordered[0], ordered[1], ordered[2])
                a3 = angle(ordered[1], ordered[2], ordered[3])
                a4 = angle(ordered[2], ordered[3], ordered[0])
                angle_err = (abs(a1 - 90) + abs(a2 - 90) + abs(a3 - 90) + abs(a4 - 90)) / 4.0
                if angle_err > 14:
                    continue

                # 平行四辺形整合性チェック (対辺の平行度)
                br_exp = ordered[3] + (ordered[1] - ordered[0])
                dist_br = np.linalg.norm(ordered[2] - br_exp)

                score = area - angle_err * 250 - dist_br * 30
                if score > best_score:
                    best_score = score
                    best_quad = ordered

            if best_quad is not None:
                # 幾何正則化: 右下頂点が指のふくらみ等で外側に突出している場合、平行四辺形補完で補正
                br_exp = best_quad[3] + (best_quad[1] - best_quad[0])
                if np.linalg.norm(best_quad[2] - br_exp) > 5.0:
                    best_quad[2] = br_exp
                return best_quad.reshape(4, 1, 2).astype(np.int32)

        return None

    def _is_touching_image_border(self, pts: np.ndarray, img_w: int, img_h: int, margin: int = 5) -> bool:
        """4頂点のうち2点以上が画像の最外縁に接しているかチェック (机の影・画像枠の誤認を排除)"""
        p = pts.reshape(-1, 2)
        touch_count = 0
        for x, y in p:
            if x <= margin or x >= img_w - margin or y <= margin or y >= img_h - margin:
                touch_count += 1
        return touch_count >= 2
