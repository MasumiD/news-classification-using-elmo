import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizer import Tokenizer
from elmo import ELMoBiLM

def sentence_to_indices(sentence, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in sentence]

class FrozenLambdaElmoClassifier(nn.Module):
    def __init__(self, elmo_model, num_classes, rnn_hidden_size=128, rnn_layers=1, fixed_lambdas=None):
        super(FrozenLambdaElmoClassifier, self).__init__()
        self.elmo = elmo_model
        for param in self.elmo.parameters():
            param.requires_grad = False

        self.embed_dim = self.elmo.embedding.embedding_dim  
        self.hidden_dim = self.elmo.lstm1.hidden_size         

        if self.embed_dim != self.hidden_dim:
            self.e0_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        else:
            self.e0_proj = nn.Identity()
        random = torch.rand(3)
        if fixed_lambdas is None:
            self.lambdas = random / random.sum()
        else:
            self.lambdas = fixed_lambdas

        self.classifier_rnn = nn.LSTM(input_size=self.hidden_dim,
                                      hidden_size=rnn_hidden_size,
                                      num_layers=rnn_layers,
                                      batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, input_ids):
        _, _, (emb, hf, hb) = self.elmo(input_ids)
        e0 = self.e0_proj(emb)
        e1 = hf
        e2 = hb
        combined = self.lambdas[0] * e0 + self.lambdas[1] * e1 + self.lambdas[2] * e2
        rnn_out, (h_n, _) = self.classifier_rnn(combined)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

class TrainableLambdaElmoClassifier(nn.Module):

    def __init__(self, elmo_model, num_classes, rnn_hidden_size=128, rnn_layers=1):
        super(TrainableLambdaElmoClassifier, self).__init__()
        self.elmo = elmo_model
        for param in self.elmo.parameters():
            param.requires_grad = False

        self.embed_dim = self.elmo.embedding.embedding_dim
        self.hidden_dim = self.elmo.lstm1.hidden_size

        if self.embed_dim != self.hidden_dim:
            self.e0_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        else:
            self.e0_proj = nn.Identity()

        self.lambda_params = nn.Parameter(torch.zeros(3))

        self.classifier_rnn = nn.LSTM(input_size=self.hidden_dim,
                                      hidden_size=rnn_hidden_size,
                                      num_layers=rnn_layers,
                                      batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, input_ids):
        _, _, (emb, hf, hb) = self.elmo(input_ids)
        e0 = self.e0_proj(emb)
        e1 = hf
        e2 = hb
        lambdas = torch.softmax(self.lambda_params, dim=0)
        combined = lambdas[0] * e0 + lambdas[1] * e1 + lambdas[2] * e2
        rnn_out, (h_n, _) = self.classifier_rnn(combined)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

class LearnableFunctionElmoClassifier(nn.Module):

    def __init__(self, elmo_model, num_classes, rnn_hidden_size=128, rnn_layers=1):
        super(LearnableFunctionElmoClassifier, self).__init__()
        self.elmo = elmo_model
        for param in self.elmo.parameters():
            param.requires_grad = False

        self.embed_dim = self.elmo.embedding.embedding_dim
        self.hidden_dim = self.elmo.lstm1.hidden_size

        if self.embed_dim != self.hidden_dim:
            self.e0_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        else:
            self.e0_proj = nn.Identity()
        
        self.mlp_combiner = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
        )

        self.classifier_rnn = nn.LSTM(input_size=self.hidden_dim,
                                      hidden_size=rnn_hidden_size,
                                      num_layers=rnn_layers,
                                      batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, input_ids):
        
        with torch.no_grad():
            _, _, (emb, hf, hb) = self.elmo(input_ids)
        e0 = self.e0_proj(emb)  
        e1 = hf               
        e2 = hb               

        
        concatenated = torch.cat([e0, e1, e2], dim=2)  
        
        combined_token = self.mlp_combiner(concatenated)  

        
        rnn_out, (h_n, _) = self.classifier_rnn(combined_token)
        last_hidden = h_n[-1]  
        logits = self.fc(last_hidden)
        return logits

class NewsDataset(Dataset):
    """
    News classification dataset that reads data from a CSV file.
    Assumes the CSV file has columns 'text' and 'label'.
    """
    def __init__(self, dataframe, vocab, tokenizer):
        self.data = dataframe
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Description']
        label = self.data.iloc[idx]['Class Index']-1
        
        tokenized_sentences = self.tokenizer.tokenize(text)
        
        tokens = [token for sentence in tokenized_sentences for token in sentence]
        indices = sentence_to_indices(tokens, self.vocab)
        return indices, label

def collate_classification(batch):
    pad_idx = 0
    max_len = max(len(x[0]) for x in batch)
    input_batch, label_batch = [], []
    for input_ids, label in batch:
        seq_len = len(input_ids)
        pad_tensor = torch.full((max_len - seq_len,), pad_idx, dtype=torch.long)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        input_batch.append(torch.cat([input_tensor, pad_tensor]))
        label_batch.append(label)
    return torch.stack(input_batch), torch.tensor(label_batch, dtype=torch.long)


def train_classifier(model, dataloader, device, epochs=3, lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_correct = 0
        total_examples = 0
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/len(dataloader):.4f}")
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)
        acc = total_correct / total_examples
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {epoch+1} finished. Loss: {avg_loss:.4f}, Training Accuracy: {acc*100:.2f}%")
    return model

def main():
    
    EMBED_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    CLASSIFIER_EPOCHS = 10
    LEARNING_RATE = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = "/kaggle/input/idk/pytorch/default/1/vocab.pkl"
    bilstm_path = "/kaggle/input/idk/pytorch/default/1/bilstm.pt"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    if not os.path.exists(bilstm_path):
        raise FileNotFoundError(f"Pre-trained ELMo model not found at {bilstm_path}")

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    elmo_model = ELMoBiLM(vocab_size, EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS)
    elmo_model.load_state_dict(torch.load(bilstm_path, map_location=device))
    elmo_model.eval()


    csv_path = "/kaggle/input/news-classification/train.csv"
    df = pd.read_csv(csv_path)
    
    tokenizer = Tokenizer()
    news_dataset = NewsDataset(df, vocab, tokenizer)
    news_dataloader = DataLoader(news_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_classification, num_workers=4, pin_memory=True)


    
    num_classes = df['Class Index'].nunique()
    classifiers = {
        
        
        "learnable": LearnableFunctionElmoClassifier(elmo_model, num_classes=num_classes)
    }

    for method_name, classifier in classifiers.items():
        print(f"\nTraining classifier with method: {method_name}")
        classifier = train_classifier(classifier, news_dataloader, device, epochs=CLASSIFIER_EPOCHS, lr=LEARNING_RATE)
        save_path = f"classifier_{method_name}.pt"
        torch.save(classifier.state_dict(), save_path)
        print(f"Saved {method_name} classifier to {save_path}")


