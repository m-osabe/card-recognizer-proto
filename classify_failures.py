"""
classify_failures.py
失敗事例を「入力起因」と「ロジック改善可能」に詳細分類するスクリプト
"""

import json
import os
import cv2
import numpy as np

with open('output/failure_analysis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

failures = data['failures']

categories = {
    'A_input_damaged': {
        'title': 'A. 入力画像起因（画像破損・激しいブレ・情報欠損）',
        'subcategories': {
            'A1_extreme_blur': {'name': 'A-1: 極端なブレ・ピンボケ（特徴点抽出が物理的に困難）', 'items': []},
            'A2_extreme_glare': {'name': 'A-2: 強烈な光反射・白飛び（カード絵柄が消失）', 'items': []},
            'A3_heavy_occlusion': {'name': 'A-3: 激しい見切れ・指遮蔽・極端な画角外', 'items': []},
            'A4_dataset_irregular': {'name': 'A-4: テストデータ異常（ラベル不一致等）', 'items': []}
        }
    },
    'B_logic_improvable': {
        'title': 'B. 判定ロジック・アルゴリズムの改良で正しくなる可能性が高いもの',
        'subcategories': {
            'B1_frame_detection_failed': {'name': 'B-1: 枠検出失敗・ズレ（画質は読めるのに枠取りが失敗）', 'items': []},
            'B2_series_twin_conflict': {'name': 'B-2: 同型カード競合（card-6 vs card-7 等の類似枠競合）', 'items': []},
            'B3_low_contrast_recoverable': {'name': 'B-3: 暗所・低コントラスト（前処理補正で抽出可能なもの）', 'items': []},
            'B4_ranking_threshold_issue': {'name': 'B-4: スコアリング・順位逆転（正解候補が上位にあるが僅差で負け）', 'items': []}
        }
    }
}

for item in failures:
    fname = item['file_name']
    gt = item['ground_truth']
    pred = item['pred_top1']
    det = item['detected']
    q = item.get('quality', {})
    lap = q.get('laplacian_var', 0)
    over = q.get('overexp_ratio', 0)
    bright = q.get('mean_brightness', 0)
    std_b = q.get('std_brightness', 0)
    gt_cand = item.get('gt_cand_info')
    
    # 分類ロジック
    # 1. テストデータ異常
    if fname.startswith('test1') or fname.startswith('test2'):
        categories['A_input_damaged']['subcategories']['A4_dataset_irregular']['items'].append(item)
    # 2. 極端なブレ (Laplacian < 25)
    elif lap < 25.0:
        categories['A_input_damaged']['subcategories']['A1_extreme_blur']['items'].append(item)
    # 3. 中等度のブレ (Laplacian 25-45) かつ 枠検出失敗
    elif lap < 45.0 and not det:
        categories['A_input_damaged']['subcategories']['A1_extreme_blur']['items'].append(item)
    # 4. 白飛び率高 (overexp > 0.08)
    elif over > 0.08:
        categories['A_input_damaged']['subcategories']['A2_extreme_glare']['items'].append(item)
    # 5. 同型カード競合 (card-6 vs card-7) で枠検出成功
    elif gt == 'card-6' and ('card-7' in pred or det):
        categories['B_logic_improvable']['subcategories']['B2_series_twin_conflict']['items'].append(item)
    # 6. 正解カードが候補リスト内に存在（Top-2〜Top-5）
    elif item['gt_in_candidates'] and gt_cand:
        categories['B_logic_improvable']['subcategories']['B4_ranking_threshold_issue']['items'].append(item)
    # 7. 枠検出失敗だが画像はブレていない (Laplacian >= 45) -> 枠検出ロジックで救済可能
    elif not det and lap >= 45.0:
        categories['B_logic_improvable']['subcategories']['B1_frame_detection_failed']['items'].append(item)
    # 8. 暗所・低コントラスト (std_b < 40 or bright < 90)
    elif std_b < 40.0 or bright < 90.0:
        categories['B_logic_improvable']['subcategories']['B3_low_contrast_recoverable']['items'].append(item)
    # 9. その他はロジック改善カテゴリへ
    else:
        categories['B_logic_improvable']['subcategories']['B4_ranking_threshold_issue']['items'].append(item)

# 集計結果の出力
print('======================================================================')
print('              失敗原因の分類集計結果 (全 92 件)')
print('======================================================================')

for cat_k, cat_v in categories.items():
    sub_tot = sum(len(sub['items']) for sub in cat_v['subcategories'].values())
    title_str = cat_v['title']
    print(f"\n## {title_str} : {sub_tot} 件 ({sub_tot/len(failures)*100:.1f}%)")
    for sub_k, sub_v in cat_v['subcategories'].items():
        cnt = len(sub_v['items'])
        pct = cnt / len(failures) * 100.0
        name_str = sub_v['name']
        print(f"   - {name_str}: {cnt} 件 ({pct:.1f}%)")
        if cnt > 0:
            sample_fnames = [x['file_name'] for x in sub_v['items'][:6]]
            print(f"     例: {', '.join(sample_fnames)}")

# 各分類の詳細分析用JSONの保存
with open('output/failure_categorized.json', 'w', encoding='utf-8') as f:
    json.dump(categories, f, ensure_ascii=False, indent=2)
print('\n[OK] 分類結果を output/failure_categorized.json に保存しました')
