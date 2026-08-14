import argparse
import sys
import os
import time
from pathlib import Path

# Windowsコンソール文字コード対応
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from card_engine import CardRecognitionEngine
from sample_generator import generate_sample_dataset


def print_banner():
    print("=" * 60)
    print("      TCG Card Recognition & Identification System")
    print("=" * 60)


def cmd_setup_demo(args, engine: CardRecognitionEngine):
    """デモ用のサンプルデータ生成とインデックス登録を一度に実行"""
    print_banner()
    print("[1/2] サンプルカードデータと撮影テスト画像を生成中...")
    generate_sample_dataset(engine.master_dir, engine.data_dir / "test")

    print("\n[2/2] マスター画像のインデックスを構築中...")
    engine.build_index()
    print("\n[OK] セットアップが完了しました！")
    print("以下のコマンドで判定テストを実行できます:")
    print("  python cli.py evaluate")
    print("  python cli.py identify data/test/test_photo_001_Flame_Dragon.jpg")


def cmd_build_index(args, engine: CardRecognitionEngine):
    """マスター画像を登録してインデックス化"""
    print_banner()
    master_dir = args.master_dir or str(engine.master_dir)
    count = engine.build_index(master_dir)
    print(f"\n合計 {count} 枚のカードをインデックス化しました。")


def cmd_identify(args, engine: CardRecognitionEngine):
    """単一の写真からカードを判定"""
    print_banner()
    if not engine.load_index():
        print("インデックスが見つかりません。先にインデックスを構築してください:")
        print("  python cli.py build-index")
        return

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"エラー: 画像ファイルが見つかりません: {img_path}")
        return

    print(f"入力画像: {img_path.name}")
    print(f"照合手法: {args.method.upper()}")
    print("-" * 60)

    start_t = time.time()
    result = engine.identify(
        str(img_path),
        method=args.method,
        top_k=args.top_k,
        save_visual_result=True,
    )
    elapsed = (time.time() - start_t) * 1000

    print(f"処理時間: {elapsed:.1f} ms")
    print(f"カード領域検出: {'[OK] 成功' if result['detected'] else '[NG] 検出できず（全体をフォールバック使用）'}")
    print("\n【判定結果 - 上位候補】")

    candidates = result.get("candidates", [])
    if not candidates:
        print("一致するカードが見つかりませんでした。")
        return

    for rank, cand in enumerate(candidates, 1):
        name = cand["name"]
        cid = cand["card_id"]
        inliers = cand.get("inliers", 0)
        sift_s = cand.get("sift_score", cand.get("score", 0.0))
        emb_s = cand.get("emb_score", 0.0)
        comb_s = cand.get("combined_score", cand.get("score", 0.0))

        star = "[*] [Top-1 Best Match]" if rank == 1 else f"    [候補 {rank}]"
        print(f"{star}")
        print(f"   カード名 : {name} (ID: {cid})")
        print(f"   総合スコア: {comb_s:.1f}点 (SIFTインライア数: {inliers}, 特徴類似度: {emb_s:.1f}%)")

    if result.get("visual_result_path"):
        print("\n[INFO] 照合結果の比較画像を保存しました:")
        print(f"   {result['visual_result_path']}")


def cmd_evaluate(args, engine: CardRecognitionEngine):
    """テストディレクトリ内の全画像を評価し、精度と速度のレポートを表示"""
    print_banner()
    if not engine.load_index():
        print("インデックスを構築中...")
        engine.build_index()

    test_dir = Path(args.test_dir) if args.test_dir else engine.data_dir / "test"
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    test_files = [f for f in test_dir.iterdir() if f.suffix.lower() in valid_exts]

    if not test_files:
        print(f"テスト画像がありません: {test_dir}")
        return

    print(f"テスト対象画像数: {len(test_files)} 枚")
    print(f"テストディレクトリ: {test_dir}")
    print("-" * 60)

    correct_top1 = 0
    correct_top3 = 0
    detected_count = 0
    total_time = 0.0

    print(f"{'ファイル名':<30} {'正解カード':<20} {'判定結果':<20} {'スコア':<8} {'結果':<6}")
    print("-" * 88)

    for fpath in test_files:
        # ファイル名から正解カードIDを推定 (test_photo_001_Flame_Dragon.jpg -> 001_Flame_Dragon)
        ground_truth = fpath.stem.replace("test_photo_", "")

        t0 = time.time()
        res = engine.identify(
            str(fpath), method=args.method, top_k=3, save_visual_result=True
        )
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

        score_str = f"{cands[0].get('combined_score', cands[0].get('score', 0)):.1f}" if cands else "0.0"
        print(f"{fpath.name[:28]:<30} {ground_truth[:18]:<20} {top1_id[:18]:<20} {score_str:<8} {mark:<6}")

    print("=" * 88)
    n = len(test_files)
    print(f"【評価レポート】")
    print(f"  ・Top-1 正解率   : {correct_top1}/{n} ({correct_top1 / n * 100:.1f}%)")
    print(f"  ・Top-3 正解率   : {correct_top3}/{n} ({correct_top3 / n * 100:.1f}%)")
    print(f"  ・カード領域検出率: {detected_count}/{n} ({detected_count / n * 100:.1f}%)")
    print(f"  ・平均処理時間   : {total_time / n * 1000:.1f} ms / 枚 (CPU)")
    print(f"  ・結果画像出力先 : {engine.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TCG Card Recognition System")
    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # setup-demo
    subparsers.add_parser("setup-demo", help="デモ用サンプルカードとテスト画像を自動生成してセットアップ")

    # build-index
    p_build = subparsers.add_parser("build-index", help="マスターカード画像を登録してインデックスを構築")
    p_build.add_argument("--master-dir", type=str, help="マスター画像フォルダのパス")

    # identify
    p_id = subparsers.add_parser("identify", help="1枚の写真を判定")
    p_id.add_argument("image", type=str, help="判定したい写真のパス")
    p_id.add_argument("--method", type=str, default="ensemble", choices=["ensemble", "sift", "embedding"], help="照合手法")
    p_id.add_argument("--top-k", type=int, default=3, help="上位候補の表示数")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="テスト画像フォルダを一括評価")
    p_eval.add_argument("--test-dir", type=str, help="テスト画像フォルダのパス")
    p_eval.add_argument("--method", type=str, default="ensemble", choices=["ensemble", "sift", "embedding"], help="照合手法")

    args = parser.parse_args()

    engine = CardRecognitionEngine(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        output_dir=os.path.join(os.path.dirname(__file__), "output"),
    )

    if args.command == "setup-demo":
        cmd_setup_demo(args, engine)
    elif args.command == "build-index":
        cmd_build_index(args, engine)
    elif args.command == "identify":
        cmd_identify(args, engine)
    elif args.command == "evaluate":
        cmd_evaluate(args, engine)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
