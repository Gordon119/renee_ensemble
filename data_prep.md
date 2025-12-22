```bash
python -W ignore -u utils/CreateTokenizedFiles.py \
--data-dir Datasets/LF-Amazon-1.3M \
--max-length 32 \
--tokenizer-type bert-base-uncased \
--tokenize-label-texts
```

To create a dataset having label-text augmentation, we can use the following command:
```bash
python utils/CreateAugData.py \
--data-dir Datasets/LF-Amazon-1.3M \
--tokenization-folder bert-base-uncased-32 \
--max-len 32
```
