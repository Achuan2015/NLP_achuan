from transformers import BertTokenizer

model_path="input/roberta_zh_large"
max_seq_len=30
batch_size=64
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(model_path)