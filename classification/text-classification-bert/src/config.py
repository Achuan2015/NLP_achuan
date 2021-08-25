from transformers import BertTokenizer

model_path="input/roberta_zh_large"
max_seq_len= 36
batch_size=128
hidden_size=768
hidden_dropout_prob=0.3
weight_decay=1e-3
epochs=3
learning_rate=2e-5
num_labels=3
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(model_path)
model_output="output/roberta_zh_large_finetuing/bert_classifier_3.bin"
model_output_dir="output/roberta_zh_large_finetuing"