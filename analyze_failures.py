"""
analyze_failures.py
全テスト画像を評価し、失敗した画像を詳細に分析・分類するスクリプト
"""

import cv2
import numpy as np
from pathlib import Path
import re
import json
import time
from collections import defaultdict
from card_engine import CardRecognitionEngine

def analyze_image_quality(img_bgr):
    """画像の物理的品質（ブレ、輝度、白飛び、黒潰れなど）を計算"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # ブラー度 (Laplacian 分散: 低いほどピンボケ/ブレ)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 輝度統計
    mean_bright = float(np.mean(gray))
    std_bright = float(np.std(gray))
    
    # 白飛び率 (輝度 > 245) と 黒潰れ率 (輝度 < 15)
    overexp_ratio = float(np.sum(gray >= 245) / (h * w))
    underexp_ratio = float(np.sum(gray <= 15) / (h * w))
    
    return {
        "width": w,
        "height": h,
        "laplacian_var": round(laplacian_var, 1),
        "mean_brightness": round(mean_bright, 1),
        "std_brightness": round(std_bright, 1),
        "overexp_ratio": round(overexp_ratio, 4),
        "underexp_ratio": round(underexp_ratio, 4),
    }

def main():
    engine = CardRecognitionEngine()
    if not engine.load_index():
        print("インデックスを構築中...")
        engine.build_index()

    test_dir = engine.data_dir / "test"
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    test_files = sorted([f for f in test_dir.iterdir() if f.suffix.lower() in valid_exts])
    
    print(f"テスト対象画像数: {len(test_files)} 枚")
    
    failures = []
    card_stats = defaultdict(lambda: {"total": 0, "correct": 0, "miss": 0})
    
    t0_all = time.time()
    for idx, fpath in enumerate(test_files):
        stem = fpath.stem
        if stem.startswith("test_photo_"):
            ground_truth = stem.replace("test_photo_", "")
        else:
            m = re.match(r"card_(\d+)_", stem)
            if m:
                ground_truth = f"card-{int(m.group(1))}"
            else:
                ground_truth = stem
        
        card_stats[ground_truth]["total"] += 1
        
        # 認識実行
        t0 = time.time()
        res = engine.identify(str(fpath), top_k=5, save_visual_result=False)
        elapsed = (time.time() - t0) * 1000
        
        detected = res.get("detected", False)
        det_method = res.get("meta", {}).get("detection_method", "none") if res.get("meta") else "none"
        candidates = res.get("candidates", [])
        
        top1 = candidates[0] if candidates else None
        top1_id = top1["card_id"] if top1 else "None"
        top1_score = top1.get("combined_score", 0.0) if top1 else 0.0
        top1_inliers = top1.get("inliers", 0) if top1 else 0
        top1_geom_valid = top1.get("is_geom_valid", False) if top1 else False
        
        top3_ids = [c["card_id"] for c in candidates[:3]]
        is_top1 = (ground_truth == top1_id) or (ground_truth in top1_id)
        is_top3 = any(ground_truth in cid or cid in ground_truth for cid in top3_ids)
        
        if is_top1:
            card_stats[ground_truth]["correct"] += 1
        else:
            card_stats[ground_truth]["miss"] += 1
            
            # 正解カードが候補の何位にいるか、正解カードの情報取得
            gt_rank = None
            gt_cand_info = None
            for rank, c in enumerate(candidates, 1):
                if ground_truth == c["card_id"] or (ground_truth in c["card_id"]):
                    gt_rank = rank
                    gt_cand_info = {
                        "rank": rank,
                        "card_id": c["card_id"],
                        "score": round(c.get("combined_score", 0.0), 1),
                        "inliers": c.get("inliers", 0),
                        "ill_inliers": c.get("ill_inliers", 0),
                        "is_geom_valid": c.get("is_geom_valid", False)
                    }
                    break
            
            # 元画像の品質分析
            img_bgr = cv2.imread(str(fpath))
            quality = analyze_image_quality(img_bgr) if img_bgr is not None else {}
            
            fail_info = {
                "file_name": fpath.name,
                "ground_truth": ground_truth,
                "pred_top1": top1_id,
                "top1_score": round(top1_score, 1),
                "top1_inliers": top1_inliers,
                "top1_geom_valid": top1_geom_valid,
                "is_top3": is_top3,
                "detected": detected,
                "detection_method": det_method,
                "gt_in_candidates": gt_rank is not None,
                "gt_cand_info": gt_cand_info,
                "quality": quality,
                "elapsed_ms": round(elapsed, 1)
            }
            failures.append(fail_info)
        
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_files):
            print(f"進捗: {idx+1}/{len(test_files)} 完了 (現在の失敗数: {len(failures)})")

    total_elapsed = time.time() - t0_all
    correct_count = len(test_files) - len(failures)
    acc = correct_count / len(test_files) * 100.0
    
    print("\n" + "=" * 60)
    print(f"全 {len(test_files)} 枚の評価完了: 正解 {correct_count} 枚, 失敗 {len(failures)} 枚 ({acc:.1f}%)")
    print(f"総処理時間: {total_elapsed:.1f} 秒 (平均: {total_elapsed / len(test_files) * 1000:.1f} ms/枚)")
    print("=" * 60)
    
    # 失敗事例をJSONに保存
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump({
            "total": len(test_files),
            "correct": correct_count,
            "accuracy": round(acc, 2),
            "card_stats": dict(card_stats),
            "failures": failures
        }, f, ensure_ascii=False, indent=2)
        
    print(f"[OK] 失敗分析データを保存しました: output/failure_analysis.json")

if __name__ == "__main__":
    main()
