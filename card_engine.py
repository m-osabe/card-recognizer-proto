"""
card_engine.py
カード認識エンジンのメインコントローラー
検出、特徴抽出、照合、アンサンブル、可視化結果の出力を統合する
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Windowsコンソール文字コード対応
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from card_detector import CardDetector
from matcher_sift import SIFTCardMatcher
from matcher_embedding import EmbeddingCardMatcher


class CardRecognitionEngine:
    def __init__(self, data_dir: str = "data", output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.master_dir = self.data_dir / "master"
        self.index_dir = self.data_dir / "index"
        self.output_dir = Path(output_dir)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.detector = CardDetector()
        self.sift_matcher = SIFTCardMatcher()
        self.emb_matcher = EmbeddingCardMatcher()

    def build_index(self, master_folder: Optional[str] = None) -> int:
        """
        マスター画像フォルダ内の全カード画像をスキャンしてインデックスを構築
        画像ファイル名（拡張子除く）がカード名/IDとして扱われます。
        例: "001_BlueEyesWhiteDragon.png", "Charizard.jpg"
        """
        folder = Path(master_folder) if master_folder else self.master_dir
        if not folder.exists():
            raise FileNotFoundError(f"マスターディレクトリが存在しません: {folder}")

        # 画像拡張子
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]

        count = 0
        print(f"[{len(image_files)} 枚] のマスター画像をインデックス化中...")

        for idx, img_path in enumerate(image_files, 1):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            card_id = img_path.stem
            # ファイル名からカード名を抽出（アンダースコアをスペースにする等）
            card_name = card_id.replace("_", " ")

            # SIFTおよびEmbeddingの両方に登録
            self.sift_matcher.register_card(card_id, card_name, img_bgr, str(img_path))
            self.emb_matcher.register_card(card_id, card_name, img_bgr, str(img_path))
            count += 1
            if idx % 20 == 0 or idx == len(image_files):
                print(f"  進捗: {idx}/{len(image_files)} 枚完了")

        # インデックスをファイルに保存
        self.sift_matcher.save_index(str(self.index_dir / "sift_index.pkl"))
        self.emb_matcher.save_index(str(self.index_dir / "emb_index.pkl"))
        print(f"[OK] インデックス構築完了: {count} 枚登録 (保存先: {self.index_dir})")
        return count

    def load_index(self) -> bool:
        """保存されたインデックスをロード"""
        sift_ok = self.sift_matcher.load_index(str(self.index_dir / "sift_index.pkl"))
        emb_ok = self.emb_matcher.load_index(str(self.index_dir / "emb_index.pkl"))
        return sift_ok and emb_ok

    def identify(
        self,
        image_input,
        method: str = "ensemble",
        top_k: int = 3,
        save_visual_result: bool = True,
        result_filename: Optional[str] = None,
    ) -> Dict:
        """
        1枚の写真からカードを検出して識別を行う。
        Args:
            image_input: 画像ファイルパス (str) または cv2画像配列 (np.ndarray)
            method: 'sift', 'embedding', 'ensemble' (デフォルト: アンサンブル)
            top_k: 返す上位候補数
            save_visual_result: 判定結果の比較画像を保存するかどうか
        Returns:
            Dict: {
                'detected': bool,
                'candidates': List[dict],
                'best_match': dict,
                'visual_result_path': Optional[str]
            }
        """
        if isinstance(image_input, (str, Path)):
            orig_img = cv2.imread(str(image_input))
            if orig_img is None:
                raise ValueError(f"画像を読み込めませんでした: {image_input}")
            input_name = Path(image_input).stem
        else:
            orig_img = image_input.copy()
            input_name = "query_image"

        # 1. カード領域の自動検出 & 透視変換
        cropped_card, corners, meta = self.detector.detect_and_crop(orig_img)

        # 2. 各マッチャーでスコアリング
        sift_results = self.sift_matcher.match(cropped_card, top_k=top_k * 2)
        emb_results = self.emb_matcher.match(cropped_card, top_k=top_k * 2)

        # 3. 手法に応じた集約
        candidates = []
        if method == "sift":
            candidates = sift_results[:top_k]
        elif method == "embedding":
            candidates = emb_results[:top_k]
        else:
            # アンサンブル: イラスト領域加重SIFTスコアと色彩類似度を組み合わせ
            scores_map = {}
            for r in sift_results:
                cid = r["card_id"]
                scores_map[cid] = {
                    "card_id": cid,
                    "name": r["name"],
                    "sift_score": r["score"],
                    "inliers": r["inliers"],
                    "ill_inliers": r.get("ill_inliers", 0),
                    "title_inliers": r.get("title_inliers", 0),
                    "emb_score": 0.0,
                    "image_path": r["image_path"],
                }
            for r in emb_results:
                cid = r["card_id"]
                if cid not in scores_map:
                    scores_map[cid] = {
                        "card_id": cid,
                        "name": r["name"],
                        "sift_score": 0.0,
                        "inliers": 0,
                        "ill_inliers": 0,
                        "title_inliers": 0,
                        "emb_score": r["score"],
                        "image_path": r["image_path"],
                    }
                else:
                    scores_map[cid]["emb_score"] = r["score"]

            for item in scores_map.values():
                sift_s = item["sift_score"]
                emb_s = item["emb_score"]
                ill_inl = item["ill_inliers"]

                # イラストインライアが10個以上ある場合は、イラスト一致の確信度ボーナス
                if ill_inl >= 10:
                    combined = sift_s * 0.5 + emb_s * 0.5 + 10.0
                else:
                    combined = sift_s * 0.3 + emb_s * 0.7

                item["combined_score"] = combined

            candidates = sorted(
                scores_map.values(),
                key=lambda x: (x.get("combined_score", 0), x.get("ill_inliers", 0)),
                reverse=True,
            )[:top_k]

        best_match = candidates[0] if candidates else None

        # 4. 可視化画像の生成
        visual_path = None
        if save_visual_result and best_match:
            visual_img = self._create_visual_report(
                orig_img, corners, cropped_card, best_match, candidates
            )
            out_name = result_filename or f"result_{input_name}.jpg"
            visual_path = str(self.output_dir / out_name)
            cv2.imwrite(visual_path, visual_img)

        return {
            "detected": meta.get("detected", False),
            "candidates": candidates,
            "best_match": best_match,
            "visual_result_path": visual_path,
        }

    def _create_visual_report(
        self,
        orig_img: np.ndarray,
        corners: Optional[np.ndarray],
        cropped_card: np.ndarray,
        best_match: dict,
        candidates: List[dict],
    ) -> np.ndarray:
        """
        写真、検出カード、正解マスター画像を横に並べた見やすい比較レポート画像を生成
        """
        # パネル1: 撮影写真（検出四角形を描画）
        annotated_orig = orig_img.copy()
        if corners is not None:
            pts = corners.astype(int)
            cv2.polylines(annotated_orig, [pts], True, (0, 255, 0), 4)
            for i, p in enumerate(pts):
                cv2.circle(annotated_orig, tuple(p), 8, (0, 0, 255), -1)

        # リサイズして高さを統一 (500px)
        h_target = 500
        w_orig = int(annotated_orig.shape[1] * (h_target / annotated_orig.shape[0]))
        p1 = cv2.resize(annotated_orig, (w_orig, h_target))

        # パネル2: 検出・補正されたカード正面画像
        w_crop = int(cropped_card.shape[1] * (h_target / cropped_card.shape[0]))
        p2 = cv2.resize(cropped_card, (w_crop, h_target))

        # パネル3: マッチしたマスター画像
        master_path = best_match.get("image_path")
        if master_path and os.path.exists(master_path):
            m_img = cv2.imread(master_path)
            w_m = int(m_img.shape[1] * (h_target / m_img.shape[0]))
            p3 = cv2.resize(m_img, (w_m, h_target))
        else:
            p3 = np.zeros((h_target, 350, 3), dtype=np.uint8)

        # ヘッダー・キャプション用余白を追加
        def add_caption(img, text, subtext=""):
            canvas = np.zeros((img.shape[0] + 60, img.shape[1], 3), dtype=np.uint8)
            canvas[60:, :] = img
            cv2.putText(
                canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            if subtext:
                cv2.putText(
                    canvas, subtext, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1
                )
            return canvas

        p1_cap = add_caption(p1, "1. Input Photo", "Detected boundary (Green)")
        p2_cap = add_caption(p2, "2. Perspective Rectified", "Aligned front view")
        score_val = best_match.get("combined_score", best_match.get("score", 0.0))
        p3_cap = add_caption(
            p3,
            f"3. Matched: {best_match['name'][:18]}",
            f"Score: {score_val:.1f} | Inliers: {best_match.get('inliers', 0)}",
        )

        # 3つのパネルを水平結合
        combined = np.hstack([p1_cap, p2_cap, p3_cap])
        return combined
