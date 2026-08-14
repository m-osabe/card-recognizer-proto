"""
matcher_sift.py
OpenCV SIFT (Scale-Invariant Feature Transform) による特徴点マッチング照合モジュール
CPUのみで高速かつ高精度にカードを同定する
"""

import cv2
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Optional


class SIFTCardMatcher:
    def __init__(self, max_features: int = 1000, ratio_threshold: float = 0.75):
        """
        max_features: 各カード画像から抽出する最大特徴点数 (500〜1500程度がバランス良好)
        ratio_threshold: Lowe's ratio test の閾値 (0.7〜0.75)
        """
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.sift = cv2.SIFT_create(nfeatures=max_features)

        # FLANN マッチャーの設定 (KDTree)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # マスターカードの特徴量キャッシュ {card_id: {'name': ..., 'kp_pts': ..., 'des': ..., 'shape': ...}}
        self.master_db: Dict[str, dict] = {}

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """画像全体から安定してSIFTキーポイントと記述子を抽出（幾何学的整合性を最大化）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # コントラスト強調（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des

    def register_card(self, card_id: str, name: str, image: np.ndarray, image_path: Optional[str] = None):
        """マスターカードを1枚登録"""
        kp, des = self.extract_features(image)
        if des is not None and len(des) > 0:
            kp_data = [(p.pt, p.size, p.angle, p.response, p.octave, p.class_id) for p in kp]
            self.master_db[card_id] = {
                "id": card_id,
                "name": name,
                "kp_data": kp_data,
                "des": des,
                "shape": image.shape,
                "image_path": image_path,
            }
            return True
        return False

    def match(
        self, query_image: np.ndarray, top_k: int = 5
    ) -> List[Dict]:
        """
        クエリ画像に対してマスターDBを検索し、イラスト・タイトルのソフト領域加重スコアで上位候補を返す。
        """
        if not self.master_db:
            return []

        q_kp, q_des = self.extract_features(query_image)
        if q_des is None or len(q_des) < 10:
            return []

        results = []

        for card_id, master_data in self.master_db.items():
            m_des = master_data["des"]
            if m_des is None or len(m_des) < 10:
                continue

            try:
                matches = self.flann.knnMatch(q_des, m_des, k=2)
            except Exception:
                continue

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

            inliers_count = 0
            ill_inliers = 0
            title_inliers = 0
            homography = None

            if len(good_matches) >= 8:
                m_kp_pts = [p[0] for p in master_data["kp_data"]]
                src_pts = np.float32([q_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([m_kp_pts[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers_count = int(np.sum(mask))
                    homography = H

                    h_m = master_data.get("shape", (840, 600))[0]
                    for idx, is_inl in enumerate(mask.ravel()):
                        if is_inl:
                            y = dst_pts[idx][0][1]
                            if 0.13 * h_m <= y <= 0.63 * h_m:
                                ill_inliers += 1
                            elif 0.04 * h_m <= y < 0.13 * h_m:
                                title_inliers += 1
            else:
                inliers_count = len(good_matches)

            # ソフト領域加重: イラスト(x3.0) + タイトル(x2.0) + 共通枠(x0.1)
            common_inliers = max(0, inliers_count - ill_inliers - title_inliers)
            weighted_inliers = ill_inliers * 3.0 + title_inliers * 2.0 + common_inliers * 0.1

            norm_factor = max(1.0, float(min(len(q_kp), len(m_des))))
            confidence = (weighted_inliers / norm_factor) * 100.0

            results.append({
                "card_id": card_id,
                "name": master_data["name"],
                "score": float(confidence),
                "inliers": inliers_count,
                "ill_inliers": ill_inliers,
                "title_inliers": title_inliers,
                "good_matches": len(good_matches),
                "image_path": master_data.get("image_path"),
                "homography": homography,
            })

        results.sort(key=lambda x: (x["score"], x["ill_inliers"]), reverse=True)
        return results[:top_k]

    def save_index(self, filepath: str):
        """マスターDBをファイルに保存"""
        with open(filepath, "wb") as f:
            pickle.dump(self.master_db, f)

    def load_index(self, filepath: str) -> bool:
        """マスターDBをファイルから読み込み"""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                self.master_db = pickle.load(f)
            return True
        return False
