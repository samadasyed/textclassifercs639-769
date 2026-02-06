import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        try:
            ckpt = torch.load(path, weights_only=False)
        except TypeError:
            ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    # Init random
    emb = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size)).astype(np.float32)
    if hasattr(vocab, "pad_id") and vocab.pad_id is not None:
        emb[vocab.pad_id] = np.zeros(emb_size, dtype=np.float32)

    # Overwrite with any pretrained vectors.
    def _iter_lines(fh):
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield line

    # Support plain text embeddings
    # if a zip file is provided, read the first file inside.
    if emb_file.endswith(".zip"):
        with zipfile.ZipFile(emb_file) as zf:
            names = zf.namelist()
            if not names:
                return emb
            with zf.open(names[0]) as f:
                for raw in f:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    parts = line.split()
                    # Skip header lines
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        continue
                    if len(parts) != emb_size + 1:
                        continue
                    word = parts[0]
                    if word in vocab:
                        try:
                            vec = np.asarray(parts[1:], dtype=np.float32)
                        except ValueError:
                            continue
                        if vec.shape[0] == emb_size:
                            emb[vocab[word]] = vec
        return emb

    with open(emb_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in _iter_lines(f):
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                continue
            if len(parts) != emb_size + 1:
                continue
            word = parts[0]
            if word in vocab:
                try:
                    vec = np.asarray(parts[1:], dtype=np.float32)
                except ValueError:
                    continue
                if vec.shape[0] == emb_size:
                    emb[vocab[word]] = vec
    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        num_words = len(self.vocab)
        embedd_size = self.args.emb_size
        hidden_size = self.args.hid_size
        hidden_layers = self.args.hid_layer
        pad_id = self.vocab.pad_id if hasattr(self.vocab, "pad_id") else None

        self.emb = nn.Embedding(num_words, embedd_size, padding_idx=pad_id)
        self.emb_dropout = nn.Dropout(self.args.emb_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()

        self.fc_layers = nn.ModuleList()
        if hidden_layers and hidden_layers > 0:
            self.fc_layers.append(nn.Linear(embedd_size, hidden_size))
            for _ in range(hidden_layers - 1):
                self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.output_layer = nn.Linear(hidden_size, self.tag_size)
        else:
            self.output_layer = nn.Linear(embedd_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = 0.08
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.uniform_(p, -v, v)
            else:
                nn.init.uniform_(p, -v, v)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        emb_tensor = torch.from_numpy(emb)
        if emb_tensor.shape != self.emb.weight.data.shape:
            raise ValueError(
                f"Embedding shape mismatch: file gives {emb_tensor.shape}, "
                f"model expects {self.emb.weight.data.shape}"
            )
        self.emb.weight.data.copy_(emb_tensor)

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        # Randomly replace words with <unk> (except <pad>)
        if self.training and self.args.word_drop > 0 and hasattr(self.vocab, "unk_id"):
            unk_id = self.vocab.unk_id
            pad_id = self.vocab.pad_id if hasattr(self.vocab, "pad_id") else None
            if unk_id is not None:
                drop_mask = torch.rand_like(x.float()) < self.args.word_drop
                if pad_id is not None:
                    drop_mask = drop_mask & (x != pad_id)
                x = x.clone()
                x[drop_mask] = unk_id

        emb = self.emb(x)  # [batch_size, seq_length, emb_size]
        emb = self.emb_dropout(emb)

        pad_id = self.vocab.pad_id if hasattr(self.vocab, "pad_id") else None
        if pad_id is not None:
            mask = (x != pad_id).float()  # [batch_size, seq_length]
            lengths = mask.sum(dim=1).clamp(min=1.0)
            mask = mask.unsqueeze(-1)
            if self.args.pooling_method == "sum":
                sent_vec = (emb * mask).sum(dim=1)
            elif self.args.pooling_method == "max":
                masked = emb.masked_fill(mask == 0, float("-inf"))
                sent_vec = masked.max(dim=1).values
                sent_vec[sent_vec == float("-inf")] = 0.0
            else:  # "avg"
                sent_vec = (emb * mask).sum(dim=1) / lengths.unsqueeze(-1)
        else:
            if self.args.pooling_method == "sum":
                sent_vec = emb.sum(dim=1)
            elif self.args.pooling_method == "max":
                sent_vec = emb.max(dim=1).values
            else:
                sent_vec = emb.mean(dim=1)

        h = sent_vec
        for layer in self.fc_layers:
            h = self.activation(layer(h))
            h = self.hid_dropout(h)
        scores = self.output_layer(h)
        return scores
