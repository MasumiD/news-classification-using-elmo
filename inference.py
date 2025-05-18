import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
from elmo import ELMoBiLM
import sys

def sentence_to_indices(sentence, vocab):
    
    return [vocab.get(token, vocab.get("<unk>", 1)) for token in sentence]

class NewsDataset(Dataset):
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

        if fixed_lambdas is None:
            self.lambdas = [1/3, 1/3, 1/3]
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
    
def load_static_embedding_model(filepath, device):
    state = torch.load(filepath, map_location=device, weights_only=True)
    embeddings_dict = state["embeddings"]
    word_to_index = state["word_to_index"]
    vocab_size = len(word_to_index)
    sample_vec = next(iter(embeddings_dict.values()))
    embed_dim = len(sample_vec)
    embedding_matrix = torch.zeros(vocab_size, embed_dim)
    for word, idx in word_to_index.items():
        vec = embeddings_dict[word] if word in embeddings_dict else [0.0] * embed_dim
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




def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <saved_model_path> <description>")
        sys.exit(1)
    
    
    saved_model_path = sys.argv[1]
    
    description = " ".join(sys.argv[2:])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  
    
    
    tokenizer = Tokenizer()
    
    
    tokenized_sentences = tokenizer.tokenize(description)
    tokens = [token for sentence in tokenized_sentences for token in sentence]
    
    
    
    if any(keyword in saved_model_path.lower() for keyword in ["frozen", "trainable", "learnable"]):
        
        
        vocab_path = "/kaggle/input/idk/pytorch/default/1/vocab.pkl"
        bilstm_path = "/kaggle/input/idk/pytorch/default/1/bilstm.pt"
        if not os.path.exists(vocab_path) or not os.path.exists(bilstm_path):
            print("Required ELMo files not found!")
            sys.exit(1)
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        vocab_size = len(vocab)
        EMBED_DIM = 100
        HIDDEN_DIM = 256
        NUM_LAYERS = 2
        
        elmo_model = ELMoBiLM(vocab_size, EMBED_DIM, HIDDEN_DIM, num_layers=NUM_LAYERS)
        elmo_model.load_state_dict(torch.load(bilstm_path, map_location=device))
        elmo_model.eval()
        
        
        model = None
        if "frozen" in saved_model_path.lower():
            model = FrozenLambdaElmoClassifier(elmo_model, num_classes=num_classes)
        elif "trainable" in saved_model_path.lower():
            model = TrainableLambdaElmoClassifier(elmo_model, num_classes=num_classes)
        elif "learnable" in saved_model_path.lower():
            model = LearnableFunctionElmoClassifier(elmo_model, num_classes=num_classes)
        else:
            print("Could not determine ELMo classifier type from filename.")
            sys.exit(1)
        
        
        state_dict = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        
        indices = sentence_to_indices(tokens, vocab)
        
    elif any(keyword in saved_model_path.lower() for keyword in ["cbow", "skipgram", "svd"]):
        
        
        static_model = None
        model = None
        if "cbow" in saved_model_path.lower():
            static_path = "/Users/masumi/Desktop/Study/Sem6/INLP/Assignments/Assignment3/models/cbow.pt"
            static_model = load_static_embedding_model(static_path, device)
            model = CBOWClassifier(static_model, num_classes=num_classes)
        elif "skipgram" in saved_model_path.lower():
            static_path = "/Users/masumi/Desktop/Study/Sem6/INLP/Assignments/Assignment3/models/skipgram.pt"
            static_model = load_static_embedding_model(static_path, device)
            model = SkipgramClassifier(static_model, num_classes=num_classes)
        elif "svd" in saved_model_path.lower():
            static_path = "/Users/masumi/Desktop/Study/Sem6/INLP/Assignments/Assignment3/models/svd.pt"
            static_model = load_static_embedding_model(static_path, device)
            model = SVDClassifier(static_model, num_classes=num_classes)
        else:
            print("Could not determine static classifier type from filename.")
            sys.exit(1)
        
        
        state_dict = torch.load(saved_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        
        vocab = static_model.vocab
        indices = sentence_to_indices(tokens, vocab)
    else:
        print("Cannot determine model type from the saved model path.")
        sys.exit(1)
    
    
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)  
    
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    
    
    for i, prob in enumerate(probs, start=1):
        print(f"class-{i} {prob:.4f}")

if __name__ == "__main__":
    main()