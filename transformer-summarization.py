from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
import torch
import evaluate
import json

# torch.autograd.set_detect_anomaly(True)

# ============================================================================================================================== #
# ============================================================================================================================== #
# ============================================================================================================================== #

print('Initializing device...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print('Loading dataset...')
# dataset = load_dataset('cnn_dailymail', '2.0.0')
dataset = load_dataset('cnn_dailymail', '3.0.0')

print('Initializing model & tokenizer...')
model_name = 'lucadiliello/bart-small'
# model_name = 'facebook/bart-base'
# model_name = 'facebook/bart-large'
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = BartTokenizer.from_pretrained(model_name)

# ============================================================================================================================== #
# ============================================================================================================================== #
# ============================================================================================================================== #


def tokenize_dataset(batch):
    inputs = batch['article']
    targets = batch['highlights']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
    summaries = tokenizer(targets, max_length=150, truncation=True, padding='max_length', return_tensors='pt')
    model_inputs['labels'] = summaries['input_ids']
    return model_inputs


print('Tokenizing dataset...')
tokenized_datasets = dataset.map(tokenize_dataset, batched=True)  # batch_size=1000

# ============================================================================================================================== #
# ============================================================================================================================== #
# ============================================================================================================================== #

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_total_limit=2,
    save_strategy='epoch',
    load_best_model_at_end=True,  # Pega a melhor das 3 epocas (observacoes indicam que mais de 3 ou 4 epocas nao adiantam muito)
)

# split definido em: https://huggingface.co/datasets/ccdv/cnn_dailymail ~92.03%/4.28%/3.68%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].shuffle().select(range(2 ** 16)),  # Approx 3h de fine-tuning
    # train_dataset=tokenized_datasets['train'].shuffle().select(range(1024)),   # Approx 15 min de fine-tuning
    eval_dataset=tokenized_datasets['validation'].shuffle(),
)

print('Fine-tuning...')
trainer.train()

# ============================================================================================================================== #
# ============================================================================================================================== #
# ============================================================================================================================== #

rouge = evaluate.load('rouge')


def generate_summary(batch):
    inputs = tokenizer(batch['article'], max_length=512, truncation=True, padding=True, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summaries = model.generate(inputs['input_ids'])
    decoded_summaries = tokenizer.batch_decode(summaries, skip_special_tokens=True)
    return decoded_summaries


def generate_summaries(tokenized_datasets):
    all_predictions = []
    all_references = []
    # batch pois nao cabe tudo na GPU (escolhida a maior potencia de 2 que ainda coube na RTX 4060 Ti)
    TEST_BATCH = 32
    LENGTH = len(tokenized_datasets['test'])
    for i in range(0, LENGTH, TEST_BATCH):
        print(f' > Generated: {i} / {LENGTH}')  # Para dar sinal de vida
        batch = tokenized_datasets['test'].select(range(i, min(i + TEST_BATCH, LENGTH)))
        references = batch['highlights']
        # ============================ #
        inputs = tokenizer(batch['article'], max_length=512, truncation=True, padding=True, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        summaries = model.generate(inputs['input_ids'])
        predictions = tokenizer.batch_decode(summaries, skip_special_tokens=True)
        # ============================ #
        all_predictions.extend(predictions)
        all_references.extend(references)
    return all_predictions, all_references


print('Generating summaries...')
predictions, references = generate_summaries(tokenized_datasets)
# ============================================================================================================================== #
# ============================================================================================================================== #
# ============================================================================================================================== #

# Eh legal de olhar
with open('PREDICTIONS.txt', 'w') as f:
    f.writelines(predictions)
    f.close()
with open('REFERENCES.txt', 'w') as f:
    f.writelines(references)
    f.close()

# Finalmente...
print('Computing ROUGE...')
result = rouge.compute(predictions=predictions, references=references)
print(json.dumps(result, indent=4))
