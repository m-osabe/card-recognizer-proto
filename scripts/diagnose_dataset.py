"""
diagnose_dataset.py
手持ち実写データセット等に対するカード検出率・Top-1/Top-3正解率を一括診断するスクリプト
"""

import os
import glob
import re
import time
import argparse
from card_engine import CardRecognitionEngine


def run_diagnosis(test_dir: str = "data/test", limit: int = None):
    engine = CardRecognitionEngine()
    if not engine.load_index():
        print("インデックスを構築中...")
        engine.build_index()

    test_files = sorted(
        glob.glob(os.path.join(test_dir, "*.jpg"))
        + glob.glob(os.path.join(test_dir, "*.png"))
    )

    if not test_files:
        print(f"テスト画像が見つかりません: {test_dir}")
        return

    if limit and limit > 0:
        test_files = test_files[:limit]

    print("=" * 88)
    print(f"  データセット診断テスト (対象: {len(test_files)} 枚)")
    print("=" * 88)
    print(f"{'ファイル名':<25} {'正解カード':<15} {'判定結果':<15} {'スコア':<8} {'検出':<6} {'結果':<6}")
    print("-" * 88)

    correct_top1 = 0
    correct_top3 = 0
    detected_count = 0
    total_time = 0.0

    for fpath in test_files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        # 正解IDの推定 (test_photo_xxx -> xxx, card_01_xxxx -> card-1)
        if stem.startswith("test_photo_"):
            ground_truth = stem.replace("test_photo_", "")
        else:
            m = re.match(r"card_(\d+)_", stem)
            if m:
                ground_truth = f"card-{int(m.group(1))}"
            else:
                ground_truth = stem

        t0 = time.time()
        res = engine.identify(fpath, method="ensemble", top_k=3, save_visual_result=True)
        t_elapsed = time.time() - t0
        total_time += t_elapsed

        if res["detected"]:
            detected_count += 1

        cands = res.get("candidates", [])
        top1_id = cands[0]["card_id"] if cands else "None"
        top3_ids = [c["card_id"] for c in cands]

        is_top1 = ground_truth == top1_id or (ground_truth in top1_id)
        is_top3 = any(ground_truth in cid or cid in ground_truth for cid in top3_ids)

        if is_top1:
            correct_top1 += 1
            mark = "OK"
        elif is_top3:
            mark = "Top-3"
        else:
            mark = "MISS"

        if is_top3:
            correct_top3 += 1

        score_str = f"{cands[0].get('combined_score', 0):.1f}" if cands else "0.0"
        det_str = "[OK]" if res["detected"] else "[NG]"
        fname = os.path.basename(fpath)
        print(f"{fname[:23]:<25} {ground_truth[:13]:<15} {top1_id[:13]:<15} {score_str:<8} {det_str:<6} {mark:<6}")

    n = len(test_files)
    print("=" * 88)
    print("【診断サマリーレポート】")
    print(f"  ・Top-1 正解率   : {correct_top1}/{n} ({correct_top1 / n * 100:.1f}%)")
    print(f"  ・Top-3 正解率   : {correct_top3}/{n} ({correct_top3 / n * 100:.1f}%)")
    print(f"  ・カード領域検出率: {detected_count}/{n} ({detected_count / n * 100:.1f}%)")
    print(f"  ・平均処理時間   : {total_time / n * 1000:.1f} ms / 枚 (CPU)")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset diagnosis script")
    parser.add_argument("--test-dir", type=str, default="data/test", help="Test image directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()
    run_diagnosis(args.test_dir, args.limit)
