## non-Label Features Datasets

AmazonTitles-670K:
```bash
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
--expname test \
--compile
```

Amazon-670K:
```bash
python main.py \
--epochs 70 \
--batch-size 256 \
--accum 1 \
--lr1 0.01 \
--lr2 4e-5 \
--warmup 10000 \
--data-dir Datasets/Amazon-670K \
--maxlen 512 \
--tf sentence-transformers/all-roberta-large-v1 \
--dropout 0.8 \
--wd1 0.001 \
--wd2 0.001 \
--noloss \
--compile
```

AmazonTitles-3M
```bash
python main.py \
--epochs 70 \
--batch-size 256 \
--lr1 0.005 \
--lr2 5e-5 \
--warmup 10000 \
--data-dir Datasets/AmazonTitles-3M \
--maxlen 32 \
--tf sentence-transformers/all-roberta-large-v1 \
--dropout 0.75 \
--wd1 0.001 \
--wd1 0.002 \
--noloss \
--compile
```

Amazon-3M
```bash
python main.py \
--epochs 60 \
--batch-size 256 \
--lr1 0.01 \
--lr2 4e-5 \
--warmup 10000 \
--data-dir Datasets/Amazon-3M \
--maxlen 256 \
--tf sentence-transformers/all-roberta-large-v1 \
--dropout 0.75 \
--wd1 0.001 \
--wd2 0.001 \
--noloss \
--compile
```

Wiki-500K
```bash
python main.py \
--epochs 100 \
--batch-size 2048 \
--lr1 0.002 \
--lr2 1e-4 \
--warmup 5000 \
--data-dir Datasets/Wiki-500K \
--maxlen 256 \
--tf sentence-transformers/paraphrase-distilroberta-base-v2 \
--dropout 0.75 \
--wd1 0.001 \
--wd2 0.001 \
--noloss \
--compile
```


## Label Features Datasets

LF-AmazonTitles-131K
```bash
python main.py \
--epochs 100 \
--batch-size 512 \
--lr1 0.05 \
--lr2 1e-5 \
--warmup 5000 \
--data-dir Datasets/LF-AmazonTitles-131K-Aug \
--maxlen 32 \
--tf sentence-transformers/msmarco-distilbert-base-v4 \
--dropout 0.85 \
--wd1 1e-4 \
--noloss \
--use-ngame-encoder ngame_pretrained_models/LF-AmazonTitles-131K-Aug/state_dict.pt \
--compile
```

LF-Wikipedia-500K
```bash
python main.py \
--epochs 100 \
--batch-size 2048 \
--lr1 0.002 \
--lr2 1e-4 \
--warmup 5000 \
--data-dir Datasets/LF-Wikipedia-500K-Aug \
--maxlen 256 \
--tf sentence-transformers/paraphrase-distilroberta-base-v2 \
--dropout 0.7 \
--wd1 0.001 \
--noloss \
--compile
```

LF-AmazonTitles-3M
```bash
python main.py \
--epochs 100 \
--batch-size 1024 \
--lr1 0.01 \
--lr2 1e-6 \
--warmup 15000 \
--data-dir Datasets/LF-AmazonTitles-1.3M-Aug \
--maxlen 32 \
--tf sentence-transformers/msmarco-distilbert-base-v4 \
--dropout 0.7 \
--wd1 1e-4 \
--noloss \
--use-ngame-encoder ngame_pretrained_models/LF-AmazonTitles-1.3M-Aug/state_dict.pt \
--compile
```


LF-WikiSeeAlso-320K
```bash
python main.py \
--epochs 100 \
--batch-size 2048 \
--lr1 0.1 \
--lr2 2e-4 \
--warmup 5000 \
--data-dir Datasets/LF-WikiSeeAlso-320K-Aug \
--maxlen 128 \
--tf sentence-transformers/paraphrase-distilroberta-base-v2 \
--dropout 0.75 \
--wd1 1e-4 \
--wd2 1e-3 \
--noloss \
--compile
```

LF-Amazon-131K
```bash
python main.py \
--epochs 100 \
--batch-size 512 \
--lr1 0.05 \
--lr2 1e-4 \
--warmup 5000 \
--data-dir Datasets/LF-Amazon-131K-Aug \
--maxlen 128 \
--tf sentence-transformers/msmarco-distilbert-base-v4 \
--dropout 0.75 \
--wd1 1e-4 \
--noloss \
--compile
```

LF-Amazon-131K | NGAME-initialisation
```bash
python main.py \
--epochs 100 \
--batch-size 512 \
--lr1 0.05 \
--lr2 1e-5 \
--warmup 5000 \
--data-dir Datasets/LF-Amazon-131K-Aug \
--maxlen 128 \
--tf sentence-transformers/msmarco-distilbert-base-v4 \
--dropout 0.85 \
--wd1 1e-4 \
--noloss \
--compile \
--use-ngame-encoder ngame_pretrained_models/LF-Amazon-131K-Aug/state_dict.pt
```







