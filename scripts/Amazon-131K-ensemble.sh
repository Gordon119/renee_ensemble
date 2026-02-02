set -e
for i in $(seq 0 99); do
  python main.py \
    --epochs 100 \
    --batch-size 512 \
    --lr1 0.05 \
    --lr2 1e-4 \
    --warmup 5000 \
    --data-dir Datasets/LF-Amazon-131K \
    --maxlen 128 \
    --tf sentence-transformers/msmarco-distilbert-base-v4 \
    --dropout 0.75 \
    --wd1 1e-4 \
    --noloss \
    --compile \
    --sample-rate 0.1 \
    --save-logits \
    --default-impl \
    --expname "$i" \
    --seed "$i"
done
