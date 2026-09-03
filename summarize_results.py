"""
summarize_results.py
全件評価結果のカード別集計スクリプト
"""
import json

with open('output/failure_analysis.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

card_stats = d['card_stats']
print('=' * 65)
print('【カード別 正答率比較】')
print('=' * 65)

for cid, st in sorted(card_stats.items(), key=lambda x: int(x[0].split('-')[1]) if '-' in x[0] else 99):
    tot = st['total']
    cor = st['correct']
    mis = st['miss']
    acc = cor / tot * 100.0 if tot > 0 else 0
    print(f'Card: {cid:<10} | 正解: {cor:2d}/{tot:2d} ({acc:5.1f}%) | 失敗: {mis:2d}')

print('=' * 65)
tot = d['total']
cor = d['correct']
acc = d['accuracy']
print(f'全体正答率: {cor}/{tot} ({acc:.1f}%) (目標 90% まであと {714 - cor} 枚)')
