import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time, re, pickle
from collections import Counter
import nltk
from nltk.corpus import brown
from nltk.tokenize import TreebankWordTokenizer, casual_tokenize
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('brown')
nltk.download('punkt')

##############################
# ELMo BiLM Model Definition (No Combine Method)
##############################
class ELMoBiLM(nn.Module):
    """
    This ELMo model does NOT combine e0, e1, e2.
    It just provides them separately:
      - e0: input embeddings
      - e1: forward hidden states
      - e2: backward hidden states
    We'll do the combination in the downstream classification step.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, pretrained_embeddings=None):
        super(ELMoBiLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm1 = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden_dim*2, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.forward_linear = nn.Linear(hidden_dim, vocab_size)
        self.backward_linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        outputs1, _ = self.lstm1(emb)      # [batch, seq_len, 2 * hidden_dim]
        outputs2, _ = self.lstm2(outputs1)
        hf = outputs2[:, :, :outputs2.size(2)//2]   # forward hidden states
        hb = outputs2[:, :, outputs2.size(2)//2:]   # backward hidden states
        # e0: emb, e1: hf, e2: hb
        forward_logits = self.forward_linear(hf)
        backward_logits = self.backward_linear(hb)
        return forward_logits, backward_logits, (emb, hf, hb)

######################################
# File: ELMO.py
######################################

##############################
# Provided Tokenizer Code
##############################
class Tokenizer:
    def __init__(self):
        self.treebank_tokenizer = TreebankWordTokenizer()
        
    def preprocess_special_cases(self, text):
        text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
        text = re.sub(r'#\w+', 'HASHTAG', text)
        text = re.sub(r'@\w+', 'MENTION', text)
        text = re.sub(r'\b\d+%|\b\d+\s?percent\b', 'PERCENTAGE', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s?(years old|yo|yrs|yr)\b', 'AGE', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)?\b', 'TIME', text)
        text = re.sub(r'\b\d+\s?(hours|hrs|minutes|mins|seconds|secs|days|weeks|months|years)\b', 'TIMEPERIOD', text, flags=re.IGNORECASE)
        return text
    
    def custom_sentence_split(self, text):
        abbreviations = [
            "Mr.", "Dr.", "Ms.", "Mrs.", "Prof.", "Sr.", "Jr.", "Ph.D.", "M.D.",
            "B.A.", "M.A.", "D.D.S.", "D.V.M.", "LL.D.", "B.C.", "a.m.", "p.m.",
            "etc.", "e.g.", "i.e.", "vs.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.",
            "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."
        ]
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace(".", "<DOT>"))
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.replace("<DOT>", ".") for s in sentences]
        return sentences

    def preprocess(self, text):
        text = self.preprocess_special_cases(text)
        sentences = self.custom_sentence_split(text)
        tokenized_sentences = []
        for sentence in sentences:
            casual_tokens = casual_tokenize(sentence, preserve_case=True)
            tokens = []
            for token in casual_tokens:
                tokens.extend(self.treebank_tokenizer.tokenize(token))
            tokenized_sentences.append(self.add_special_tokens(tokens))
        return tokenized_sentences

    def add_special_tokens(self, tokens):
        return ['START'] + tokens + ['END']

    def tokenize(self, text):
        return self.preprocess(text)

##############################
# Vocabulary and Utility Functions
##############################
def build_vocab(tokenized_sentences, max_vocab_size=20000, min_freq=1):
    """
    Builds a vocabulary from the tokenized sentences, with optional frequency filtering.
    Returns a dict: token -> index
    """
    counter = Counter()
    for sentence in tokenized_sentences:
        counter.update(sentence)
    most_common = counter.most_common(max_vocab_size)
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for token, freq in most_common:
        if freq >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
    return vocab

def sentence_to_indices(sentence, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in sentence]

##############################
# Brown Dataset for Language Modeling
##############################
class BrownDataset(Dataset):
    def __init__(self, sentences_indices):
        self.data = sentences_indices  # list of lists of token indices
        self.pad_idx = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        # For forward LM: input = tokens[:-1], target = tokens[1:]
        input_forward = tokens[:-1]
        target_forward = tokens[1:]
        # For backward LM: input = tokens[1:], target = tokens[:-1]
        input_backward = tokens[1:]
        target_backward = tokens[:-1]
        return (
            torch.tensor(input_forward, dtype=torch.long),
            torch.tensor(target_forward, dtype=torch.long),
            torch.tensor(input_backward, dtype=torch.long),
            torch.tensor(target_backward, dtype=torch.long)
        )

def collate_fn(batch):
    """
    Collate function to pad sequences within a batch to the same length.
    """
    pad_idx = 0
    max_len = max(len(x[0]) for x in batch)
    input_f_batch, target_f_batch = [], []
    input_b_batch, target_b_batch = [], []
    for (input_f, target_f, input_b, target_b) in batch:
        seq_len = len(input_f)
        pad_tensor = torch.full((max_len - seq_len,), pad_idx, dtype=torch.long)
        input_f_batch.append(torch.cat([input_f, pad_tensor]))
        target_f_batch.append(torch.cat([target_f, pad_tensor]))
        
        seq_len_b = len(input_b)
        pad_tensor_b = torch.full((max_len - seq_len_b,), pad_idx, dtype=torch.long)
        input_b_batch.append(torch.cat([input_b, pad_tensor_b]))
        target_b_batch.append(torch.cat([target_b, pad_tensor_b]))

    return (
        torch.stack(input_f_batch),
        torch.stack(target_f_batch),
        torch.stack(input_b_batch),
        torch.stack(target_b_batch)
    )


##############################
# Training Routine for ELMo
##############################
def train_elmo_model(model, dataloader, device, epochs, lr):
    """
    Trains ELMo on a forward+backward language modeling objective.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for i, (input_f, target_f, input_b, target_b) in pbar:
            input_f = input_f.to(device)
            target_f = target_f.to(device)
            input_b = input_b.to(device)
            target_b = target_b.to(device)

            optimizer.zero_grad()
            forward_logits, backward_logits, _ = model(input_f)
            loss_f = criterion(forward_logits.view(-1, forward_logits.size(-1)), target_f.view(-1))
            loss_b = criterion(backward_logits.view(-1, backward_logits.size(-1)), target_b.view(-1))
            loss = (loss_f + loss_b) / 2.0

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss/(i+1):.4f}")

        print(f"Epoch {epoch+1} completed in {time.time()-start_time:.2f}s")

    return model

##############################
# Main Function (ELMo Training)
##############################
def main():
    # Hyperparameters
    EMBED_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    MAX_VOCAB_SIZE = 20000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Brown corpus
    brown_sentences = brown.sents()  # list of lists of words
    brown_text_sentences = [" ".join(sentence) for sentence in brown_sentences]

    # Tokenize
    tokenizer = Tokenizer()
    tokenized_sentences = []
    print("Tokenizing Brown corpus...")
    for text in tqdm(brown_text_sentences, desc="Tokenizing"):
        tokenized = tokenizer.tokenize(text)
        tokenized_sentences.extend(tokenized)

    # Build vocab
    vocab = build_vocab(tokenized_sentences, max_vocab_size=MAX_VOCAB_SIZE)
    print(f"Vocabulary size: {len(vocab)}")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Convert to indices
    sentences_indices = [sentence_to_indices(sentence, vocab) for sentence in tokenized_sentences]

    # Dataset + Dataloader
    dataset = BrownDataset(sentences_indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Define and train ELMo model (no combination logic here)
    model = ELMoBiLM(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    )
    model = train_elmo_model(model, dataloader, device, EPOCHS, LEARNING_RATE)

    # Save the single ELMo model
    torch.save(model.state_dict(), "bilstm.pt")
    print("ELMo model saved to bilstm.pt")

if __name__ == "__main__":
    main()
