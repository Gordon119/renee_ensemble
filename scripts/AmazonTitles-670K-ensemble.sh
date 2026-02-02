set -e
for i in $(seq 0 99); do
  python main.py \
    --epochs 60 \
    --batch-size 256 \
    --lr1 0.001 \
    --lr2 5e-5 \
    --warmup 10000 \
    --data-dir Datasets/AmazonTitles-670K \
    --maxlen 32 \
    --tf sentence-transformers/all-roberta-large-v1 \
    --dropout 0.8 \
    --wd1 0.001 \
    --wd2 0.001 \
    --noloss \
    --compile \
    --sample-rate 0.1 \
    --save-logits \
    --default-impl \
    --expname "$i" \
    --seed "$i"
done
