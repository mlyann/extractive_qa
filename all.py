import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_metric

# Step 1: Load the SQuAD 1.1 dataset
dataset = load_dataset("squad")

from transformers import BertTokenizerFast, BertForQuestionAnswering

# Initialize the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)  # Use BertTokenizerFast
model = BertForQuestionAnswering.from_pretrained(model_name)


# Note: Compare the scores with Table 5 in the SQuAD 1.1 paper and include your observations in the notebook.

def prepare_features(examples):
    """
    Tokenize the dataset and map answers to start and end positions.
    """
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["START"] = []
    tokenized_examples["END"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["START"].append(cls_index)
            tokenized_examples["END"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["START"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["END"].append(token_end_index + 1)
            else:
                tokenized_examples["START"].append(cls_index)
                tokenized_examples["END"].append(cls_index)

    return tokenized_examples

# Tokenize dataset
tokenized_squad = dataset.map(prepare_features, batched=True, remove_columns=dataset["train"].column_names)


def create_tensor_dataset(tokenized_data):
    return TensorDataset(
        torch.tensor(tokenized_data["input_ids"], dtype=torch.long),
        torch.tensor(tokenized_data["attention_mask"], dtype=torch.long),
        torch.tensor(tokenized_data["start_positions"], dtype=torch.long),
        torch.tensor(tokenized_data["end_positions"], dtype=torch.long)
    )

train_dataset = create_tensor_dataset(tokenized_squad["train"])
eval_dataset = create_tensor_dataset(tokenized_squad["validation"])





train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16)
optimizer = AdamW(model.parameters(), lr=5e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

epochs = 18
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Unpack the batch
        input_ids, attention_mask, start_positions, end_positions = [b.to(device) for b in batch]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        # Compute loss
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")



metric = load_metric("squad")

model.eval()
all_predictions = []
all_references = []

with torch.no_grad():
    for batch in eval_loader:
        input_ids, attention_mask, start_positions, end_positions = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Convert logits to predictions
        start_predictions = torch.argmax(start_logits, dim=1)
        end_predictions = torch.argmax(end_logits, dim=1)

        for i in range(input_ids.size(0)):
            # Convert token indices back to text
            input_id_list = input_ids[i].tolist()
            pred_start = start_predictions[i].item()
            pred_end = end_predictions[i].item()

            predicted_answer = tokenizer.decode(input_id_list[pred_start:pred_end + 1], skip_special_tokens=True)
            
            # Ground truth answers
            example = dataset["validation"][i]
            references = example["answers"]["text"]

            all_predictions.append(predicted_answer)
            all_references.append(references)

# EM
results = metric.compute(predictions=all_predictions, references=all_references)
exact_match = results["exact_match"]

print(f"Exact Match (EM) Score: {exact_match:.2f}")
