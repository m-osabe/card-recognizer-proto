"""
matcher_embedding.py
画像特徴ベクトル（Embedding）および色・テクスチャ記述子による類似度検索モジュール
GPU不要・CPUのみで高速に動作し、カード全体の色彩・構図・イラストの類似度をコサイン類似度で照合する
"""

import cv2
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Optional


class GlobalFeatureExtractor:
    """
    CPU環境で超高速かつ高精度にカード全体の色彩・構図・テクスチャを多次元ベクトル化する抽出器。
    ・ブロック分割HSVカラーヒストグラム (色彩と配置)
    ・局所勾配方向ヒストグラム (HOG的エッジテクスチャ)
    ・明度コントラストモーメント
    """
    def __init__(self, grid_x: int = 4, grid_y: int = 6):
        self.grid_x = grid_x
        self.grid_y = grid_y

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 外周 10% のセーフティクロップ（机の木目・スリーブ枠・背景ノイズを完全遮断）
        h, w = image.shape[:2]
        pad_y = max(1, int(h * 0.10))
        pad_x = max(1, int(w * 0.10))
        cropped = image[pad_y : h - pad_y, pad_x : w - pad_x]
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            cropped = image

        # 正規化サイズ (240x336)
        img_resized = cv2.resize(cropped, (240, 336))
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        features = []
        cell_h = 336 // self.grid_y
        cell_w = 240 // self.grid_x

        # 1. 各グリッドセルごとのHSVカラーヒストグラム（空間的配置を保持）
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                cell_hsv = hsv[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
                # H (16ビン), S (8ビン), V (8ビン)
                hist_h = cv2.calcHist([cell_hsv], [0], None, [16], [0, 180])
                hist_s = cv2.calcHist([cell_hsv], [1], None, [8], [0, 256])
                hist_v = cv2.calcHist([cell_hsv], [2], None, [8], [0, 256])

                cv2.normalize(hist_h, hist_h)
                cv2.normalize(hist_s, hist_s)
                cv2.normalize(hist_v, hist_v)

                features.extend(hist_h.flatten())
                features.extend(hist_s.flatten())
                features.extend(hist_v.flatten())

        # 2. カード中央（イラスト領域）のディテールテクスチャ (Sobelエッジ強度・方向)
        center_gray = gray[40:200, 20:220]
        gx = cv2.Sobel(center_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(center_gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist_edge = cv2.calcHist([angle], [0], None, [16], [0, 360])
        cv2.normalize(hist_edge, hist_edge)
        features.extend(hist_edge.flatten() * 2.0)

        # L2正規化
        vec = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec


class EmbeddingCardMatcher:
    def __init__(self):
        self.extractor = GlobalFeatureExtractor()
        self.card_ids: List[str] = []
        self.card_names: List[str] = []
        self.card_paths: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D) 行列

    def register_card(self, card_id: str, name: str, image: np.ndarray, image_path: Optional[str] = None):
        """マスターカードの特徴ベクトルを抽出して登録"""
        feat = self.extractor.extract(image)
        self.card_ids.append(card_id)
        self.card_names.append(name)
        self.card_paths.append(image_path or "")

        if self.embeddings is None:
            self.embeddings = feat.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, feat])

    def match(self, query_image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        クエリ画像の特徴ベクトルとマスター全件の内積（コサイン類似度）を計算
        """
        if self.embeddings is None or len(self.card_ids) == 0:
            return []

        q_feat = self.extractor.extract(query_image)
        # コサイン類似度（L2正規化済みベクトルの内積）
        similarities = np.dot(self.embeddings, q_feat)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx]) * 100.0  # 0〜100%
            results.append({
                "card_id": self.card_ids[idx],
                "name": self.card_names[idx],
                "score": score,
                "image_path": self.card_paths[idx],
            })

        return results

    def save_index(self, filepath: str):
        data = {
            "card_ids": self.card_ids,
            "card_names": self.card_names,
            "card_paths": self.card_paths,
            "embeddings": self.embeddings,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, filepath: str) -> bool:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.card_ids = data["card_ids"]
            self.card_names = data["card_names"]
            self.card_paths = data["card_paths"]
            self.embeddings = data["embeddings"]
            return True
        return False
