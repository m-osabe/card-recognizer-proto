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
    def __init__(self, max_features: int = 800, ratio_threshold: float = 0.75):
        """
        max_features: 各カード画像から抽出する最大特徴点数 (600〜800程度で高速かつ高精度)
        ratio_threshold: Lowe's ratio test の閾値 (0.7〜0.75)
        """
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.sift = cv2.SIFT_create(nfeatures=max_features)

        # マスターカードの特徴量キャッシュ
        self.master_db: Dict[str, dict] = {}
        
        # 統合FLANNインデックス用キャッシュ
        self.flann_index: Optional[cv2.flann_Index] = None
        self.all_des_matrix: Optional[np.ndarray] = None
        self.des_to_card_idx: Optional[np.ndarray] = None
        self.card_id_list: List[str] = []
        self._index_built = False

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """画像全体から安定してSIFTキーポイントと記述子を抽出"""
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
            self._index_built = False
            return True
        return False

    def build_unified_flann_index(self):
        """全マスターカードの特徴点を1つの巨大KD-Treeインデックスに一括統合 (Fast SIFT Voting用)"""
        if not self.master_db or self._index_built:
            return

        all_des = []
        des_to_card_idx = []
        self.card_id_list = []

        for idx, (cid, data) in enumerate(self.master_db.items()):
            des = data["des"]
            if des is not None and len(des) > 0:
                all_des.append(des)
                des_to_card_idx.extend([idx] * len(des))
                self.card_id_list.append(cid)

        if all_des:
            self.all_des_matrix = np.vstack(all_des).astype(np.float32)
            self.des_to_card_idx = np.array(des_to_card_idx, dtype=np.int32)
            FLANN_INDEX_KDTREE = 1
            self.flann_index = cv2.flann_Index(
                self.all_des_matrix, dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            )
            self._index_built = True

    def match(
        self,
        query_image: np.ndarray,
        top_k: int = 5,
        target_card_ids: Optional[List[str]] = None,
        coarse_top_n: int = 12,
    ) -> List[Dict]:
        """
        Fast SIFT Voting による超高速・背景耐性照合:
        1. 統合KD-Treeで全マスターに対する特徴点マッチングを1回で実行 (数ms〜数十ms)
        2. 背景ノイズに埋もれず、カード部分の特徴点投票数が多い上位候補 (coarse_top_n) を選出
        3. 上位候補に対してのみ RANSAC Homography 幾何検証を実行 (数百ms)
        """
        if not self.master_db:
            return []

        q_kp, q_des = self.extract_features(query_image)
        if q_des is None or len(q_des) < 10:
            return []

        # 統合インデックスの準備
        self.build_unified_flann_index()

        # Step 1: 統合FLANNインデックスによる一括 k-NN 探索 (k=2)
        indices, dists = self.flann_index.knnSearch(
            q_des.astype(np.float32), 2, params=dict(checks=50)
        )

        # Step 2: Lowe's Ratio Test & カード別マッチング蓄積
        matches_by_card: Dict[str, List[Tuple[int, int]]] = {}
        votes_by_card: Dict[str, int] = {}

        for q_idx in range(len(q_des)):
            d1, d2 = dists[q_idx][0], dists[q_idx][1]
            if d1 < self.ratio_threshold * d2:
                m_global_des_idx = indices[q_idx][0]
                c_idx = self.des_to_card_idx[m_global_des_idx]
                cid = self.card_id_list[c_idx]

                if target_card_ids is not None and cid not in target_card_ids:
                    continue

                votes_by_card[cid] = votes_by_card.get(cid, 0) + 1
                if cid not in matches_by_card:
                    matches_by_card[cid] = []
                matches_by_card[cid].append((q_idx, m_global_des_idx))

        if not votes_by_card:
            return []

        # 投票数上位候補を選出 (Coarse SIFT Selection)
        sorted_candidates = sorted(votes_by_card.items(), key=lambda x: x[1], reverse=True)[
            : max(coarse_top_n, top_k * 2)
        ]

        # Step 3: 上位候補に対してのみ RANSAC 幾何検証を実行 (Fine Verification)
        results = []

        # 各カードのマスターデータ内でのローカルインデックス算出用マップ
        for card_id, vote_count in sorted_candidates:
            master_data = self.master_db[card_id]
            m_kp_data = master_data["kp_data"]
            m_des = master_data["des"]

            # カード固有の単体マッチングを実行して確実なキーポイント対応を作成
            # 投票で絞り込まれた上位候補のみなので超高速 (数ms/カード)
            flann_local = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=4), dict(checks=32)
            )
            try:
                local_matches = flann_local.knnMatch(q_des, m_des, k=2)
            except Exception:
                continue

            good_matches = []
            for pair in local_matches:
                if len(pair) == 2 and pair[0].distance < self.ratio_threshold * pair[1].distance:
                    good_matches.append(pair[0])

            inliers_count = 0
            ill_inliers = 0
            title_inliers = 0
            homography = None

            if len(good_matches) >= 6:
                m_kp_pts = [p[0] for p in m_kp_data]
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

            # ソフト領域加重
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
                "votes": vote_count,
                "image_path": master_data.get("image_path"),
                "homography": homography,
            })

        results.sort(key=lambda x: (x["inliers"], x["score"]), reverse=True)
        return results[:top_k]

    @staticmethod
    def estimate_corners_from_homography(
        master_shape: Tuple[int, int, int], homography: np.ndarray, scale: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        マスター画像の形状とホモグラフィ行列 H (query -> master) から、
        クエリ画像（写真）上のカード4頂点座標を逆算して返す。
        scale: クエリ画像がリサイズされていた場合の逆スケール係数 (orig_size / query_size)
        Returns:
            np.ndarray of shape (4, 2) with [x, y] coordinates in original photo, or None
        """
        if homography is None:
            return None

        h_m, w_m = master_shape[:2]
        # マスター画像の4隅 [左上, 右上, 右下, 左下]
        master_corners = np.array(
            [[0, 0], [w_m - 1, 0], [w_m - 1, h_m - 1], [0, h_m - 1]],
            dtype="float32",
        ).reshape(-1, 1, 2)

        try:
            # H は query -> master なので、逆行列 H_inv は master -> query
            h_inv = np.linalg.inv(homography)
            query_corners = cv2.perspectiveTransform(master_corners, h_inv)
            corners = query_corners.reshape(4, 2) * scale

            # 幾何学的な妥当性検証
            approx_int = corners.astype(np.int32).reshape(-1, 1, 2)
            if not cv2.isContourConvex(approx_int):
                return None

            area = cv2.contourArea(approx_int)
            if area < 5000:  # 面積が極小（退化）
                return None

            # 4辺の長さをチェックし、極端に細長かったり歪んでいないか確認
            p = corners
            side_lens = [
                np.linalg.norm(p[0] - p[1]),
                np.linalg.norm(p[1] - p[2]),
                np.linalg.norm(p[2] - p[3]),
                np.linalg.norm(p[3] - p[0]),
            ]
            w_avg = (side_lens[0] + side_lens[2]) / 2.0
            h_avg = (side_lens[1] + side_lens[3]) / 2.0

            if w_avg < 20 or h_avg < 20:
                return None

            aspect = min(w_avg, h_avg) / max(w_avg, h_avg)
            if aspect < 0.35 or aspect > 0.95:  # TCGの標準縦横比 (約0.7) から極端に乖離
                return None

            return corners
        except Exception:
            return None

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
