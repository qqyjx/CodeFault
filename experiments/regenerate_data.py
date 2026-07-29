#!/usr/bin/env python3
"""重新生成带有正负样本的数据集"""

import json
import random
from pathlib import Path
from datetime import datetime

random.seed(42)

# 使用相对路径
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "processed" / "codefault"

# 读取原始数据
with open(DATA_DIR / "train.json") as f:
    train_data = json.load(f)
with open(DATA_DIR / "val.json") as f:
    val_data = json.load(f)
with open(DATA_DIR / "test.json") as f:
    test_data = json.load(f)

def add_negative_samples(data, neg_ratio=0.5):
    """添加负样本: 随机打乱代码-日志配对"""
    n = len(data)
    n_neg = int(n * neg_ratio)
    
    # 保留正样本
    positive = data.copy()
    for item in positive:
        item['label'] = 1
    
    # 生成负样本: 打乱配对
    negative = []
    log_contexts = [d['log_context'] for d in data]
    random.shuffle(log_contexts)
    
    for i in range(n_neg):
        item = data[i % n].copy()
        item['log_context'] = log_contexts[(i + n//2) % n]  # 错误配对
        item['label'] = 0
        item['id'] = item['id'] + '_neg'
        negative.append(item)
    
    # 合并并打乱
    combined = positive + negative
    random.shuffle(combined)
    return combined

# 重新生成
train_new = add_negative_samples(train_data, neg_ratio=0.5)
val_new = add_negative_samples(val_data, neg_ratio=0.5)
test_new = add_negative_samples(test_data, neg_ratio=0.5)

# 保存
for name, data in [('train', train_new), ('val', val_new), ('test', test_new)]:
    with open(DATA_DIR / f"{name}.json", 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    pos = sum(1 for d in data if d['label'] == 1)
    neg = len(data) - pos
    print(f"{name}: {len(data)} samples (pos={pos}, neg={neg})")

# 更新metadata
metadata = {
    'total_samples': len(train_new) + len(val_new) + len(test_new),
    'train_samples': len(train_new),
    'val_samples': len(val_new),
    'test_samples': len(test_new),
    'positive_ratio': 0.67,
    'created_at': datetime.now().isoformat()
}
with open(DATA_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n数据集重新生成完成!")
