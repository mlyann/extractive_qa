import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

# Step 1: Load the SQuAD 1.1 dataset
dataset = load_dataset("squad")

from transformers import BertTokenizerFast, BertForQuestionAnswering

# Initialize the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)  # Use BertTokenizerFast
model = BertForQuestionAnswering.from_pretrained(model_name)


# Note: Compare the scores with Table 5 in the SQuAD 1.1 paper and include your observations in the notebook.


# Step 3: Data Preparation
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

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
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
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
            else:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)

    return tokenized_examples

# Tokenize dataset
tokenized_squad = dataset.map(prepare_features, batched=True, remove_columns=dataset["train"].column_names)




# Step 4: Convert tokenized data into PyTorch Tensors
def create_tensor_dataset(tokenized_data):
    return TensorDataset(
        torch.tensor(tokenized_data["input_ids"], dtype=torch.long),
        torch.tensor(tokenized_data["attention_mask"], dtype=torch.long),
        torch.tensor(tokenized_data["start_positions"], dtype=torch.long),
        torch.tensor(tokenized_data["end_positions"], dtype=torch.long)
    )

train_dataset = create_tensor_dataset(tokenized_squad["train"])
eval_dataset = create_tensor_dataset(tokenized_squad["validation"])





# Step 5: DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=16)

# Step 6: Optimizer setup
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 7: Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Step 8: Training loop
epochs = 100
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


# Step 9: Evaluation loop
model.eval()
y_true = []
y_pred = []

for batch in eval_loader:
    input_ids, attention_mask, start_positions, end_positions = [b.to(device) for b in batch]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        start_preds = torch.argmax(outputs.start_logits, dim=-1)
        end_preds = torch.argmax(outputs.end_logits, dim=-1)

    y_true.extend(list(zip(start_positions.cpu().numpy(), end_positions.cpu().numpy())))
    y_pred.extend(list(zip(start_preds.cpu().numpy(), end_preds.cpu().numpy())))

# Step 10: Compute Exact Match (EM) and F1 Scores
def compute_metrics(y_true, y_pred):
    exact_match = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]) / len(y_true)
    print(f"Exact Match (EM): {exact_match:.4f}")
    # Add F1 calculation if needed

compute_metrics(y_true, y_pred)
