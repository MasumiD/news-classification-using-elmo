import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import time, re
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import TreebankWordTokenizer, casual_tokenize
from tqdm import tqdm
import nltk


def load_static_embedding_model(filepath, device):
    """
    Loads a static embedding model saved by svd.py, cbow.py, or skipgram.py.
    Expects the saved dictionary to have keys:
      - "embeddings": a dict mapping word to its embedding (list)
      - "word_to_index": a dict mapping word to its index
      - "index_to_word": a dict mapping index to word
    Constructs an nn.Embedding layer (with frozen weights) and returns a wrapper module.
    If a word is missing in the embeddings dictionary (e.g. "<unk>"),
    a zero vector is used.
    """
    state = torch.load(filepath, map_location=device, weights_only=True)
    embeddings_dict = state["embeddings"]
    word_to_index = state["word_to_index"]
    vocab_size = len(word_to_index)
    
    sample_vec = next(iter(embeddings_dict.values()))
    embed_dim = len(sample_vec)
    
    embedding_matrix = torch.zeros(vocab_size, embed_dim)
    for word, idx in word_to_index.items():
        if word in embeddings_dict:
            vec = embeddings_dict[word]
        else:
            
            vec = [0.0] * embed_dim
        embedding_matrix[idx] = torch.tensor(vec, dtype=torch.float)
    embedding_layer = nn.Embedding(vocab_size, embed_dim)
    embedding_layer.weight.data.copy_(embedding_matrix)
    embedding_layer.weight.requires_grad = False  

    
    class StaticEmbeddingWrapper(nn.Module):
        def __init__(self, embedding, vocab):
            super().__init__()
            self.embedding = embedding
            self.vocab = vocab
    return StaticEmbeddingWrapper(embedding_layer, word_to_index).to(device)


def sentence_to_indices(sentence, vocab):
    
    return [vocab.get(token, vocab.get("<unk>", 1)) for token in sentence]

class NewsDataset(Dataset):
    """
    News classification dataset reading from a CSV file.
    Assumes the CSV file has columns 'text' and 'label'.
    Uses the given tokenizer and vocabulary.
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


class CBOWClassifier(nn.Module):
    def __init__(self, static_model, num_classes, rnn_hidden_size=128, rnn_layers=1):
        super(CBOWClassifier, self).__init__()
        self.embedding = static_model.embedding
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.vocab = static_model.vocab
        embed_dim = self.embedding.embedding_dim
        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
    
    def forward(self, input_ids):
        emb = self.embedding(input_ids)  
        _, (h_n, _) = self.rnn(emb)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

class SkipgramClassifier(nn.Module):
    def __init__(self, static_model, num_classes, rnn_hidden_size=128, rnn_layers=1):
        super(SkipgramClassifier, self).__init__()
        self.embedding = static_model.embedding
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.vocab = static_model.vocab
        embed_dim = self.embedding.embedding_dim
        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
    
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        _, (h_n, _) = self.rnn(emb)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

class SVDClassifier(nn.Module):
    def __init__(self, static_model, num_classes, rnn_hidden_size=128, rnn_layers=1):
        super(SVDClassifier, self).__init__()
        self.embedding = static_model.embedding
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.vocab = static_model.vocab
        embed_dim = self.embedding.embedding_dim
        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)
    
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        _, (h_n, _) = self.rnn(emb)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits

def train_classifier(model, dataloader, device, epochs=3, lr=1e-3):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Classifier Training Epoch {epoch+1}")
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
    return model
