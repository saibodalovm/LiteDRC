#!/usr/bin/env bash
set -e

CKPT_DIR="ckpt/train_zero_shot"
RES_DIR="results"
DATA_ROOT="/mnt/j/processed_dataset"

mkdir -p "$CKPT_DIR" "$RES_DIR"

echo "=========================================="
echo "1/3. Training Concat models (both PDKs)"
echo "=========================================="
python3 scripts/train_zero_shot.py \
  --train-pdk sky130 \
  --test-pdk gf180 \
  --cond concat \
  --data-root "$DATA_ROOT" \
  --out-dir "$CKPT_DIR"

python3 scripts/train_zero_shot.py \
  --train-pdk gf180 \
  --test-pdk sky130 \
  --cond concat \
  --data-root "$DATA_ROOT" \
  --out-dir "$CKPT_DIR"

echo "=========================================="
echo "2/3. Evaluating In-Domain & Zero-Shot (4 runs)"
echo "=========================================="
ZS_CSV="$RES_DIR/concat_zero_shot_metrics.csv"
rm -f "$ZS_CSV"

# 1) Sky130 In-Domain (Table 3)
python3 scripts/eval_zero_shot.py \
  --train-pdk sky130 \
  --test-pdk sky130 \
  --cond concat \
  --weights "$CKPT_DIR/best_model_zeroshot_train_sky130_test_gf180_concat.pth" \
  --data-root "$DATA_ROOT" \
  --out-csv "$ZS_CSV"

# 2) Sky130 -> GF180 Zero-Shot (Table 5)
python3 scripts/eval_zero_shot.py \
  --train-pdk sky130 \
  --test-pdk gf180 \
  --cond concat \
  --weights "$CKPT_DIR/best_model_zeroshot_train_sky130_test_gf180_concat.pth" \
  --data-root "$DATA_ROOT" \
  --out-csv "$ZS_CSV"

# 3) GF180 In-Domain (Table 3)
python3 scripts/eval_zero_shot.py \
  --train-pdk gf180 \
  --test-pdk gf180 \
  --cond concat \
  --weights "$CKPT_DIR/best_model_zeroshot_train_gf180_test_sky130_concat.pth" \
  --data-root "$DATA_ROOT" \
  --out-csv "$ZS_CSV"

# 4) GF180 -> Sky130 Zero-Shot (Table 5)
python3 scripts/eval_zero_shot.py \
  --train-pdk gf180 \
  --test-pdk sky130 \
  --cond concat \
  --weights "$CKPT_DIR/best_model_zeroshot_train_gf180_test_sky130_concat.pth" \
  --data-root "$DATA_ROOT" \
  --out-csv "$ZS_CSV"

echo "=========================================="
echo "3/3. Running Few-Shot (5%) in both directions"
echo "=========================================="
python3 scripts/train_few_shot.py \
  --source-pdk sky130 \
  --target-pdk gf180 \
  --ckpt-dir "$CKPT_DIR" \
  --models Concat \
  --fractions 0.05 \
  --data-root "$DATA_ROOT" \
  --out-csv "$RES_DIR/concat_few_shot_sky130_to_gf180.csv"

python3 scripts/train_few_shot.py \
  --source-pdk gf180 \
  --target-pdk sky130 \
  --ckpt-dir "$CKPT_DIR" \
  --models Concat \
  --fractions 0.05 \
  --data-root "$DATA_ROOT" \
  --out-csv "$RES_DIR/concat_few_shot_gf180_to_sky130.csv"

echo "=========================================="
echo "ALL DONE! CSV files saved in $RES_DIR/"
echo "=========================================="