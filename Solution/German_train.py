import pandas as pd
import numpy as np
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class FocusedEmbTransform(nn.Module):
    def __init__(self, d, hidden_dim=1024):
        super(FocusedEmbTransform, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d)
        )
        self.attn_weights = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x = self.mlp(x) 
        x = x * self.attn_weights
        return x

def safe_str(x):
    if pd.isna(x):
        return ''
    return str(x)


def distance_cosine(a, b):
    cos = nn.functional.cosine_similarity(a, b, dim=1)
    return 1.0 - cos

def main():
    device = torch.device("cuda")
    embed_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2', 
                                      device=device)
    print(f"SentenceTransformer loaded. Device: {embed_model.device}")

    german_csv_path = "/path to German database" 
    df_german = pd.read_csv(german_csv_path, low_memory=False)
    print("German dataset loaded. Rows:", len(df_german))

    subject_names_corpus = df_german['Name'].fillna('').astype(str).tolist()
    corpus_embeddings_list = []
    batch_size_corpus = 64
    for i in tqdm(range(0, len(subject_names_corpus), batch_size_corpus),
                  desc="Embedding Subject Corpus", unit="batch"):
        batch_texts = subject_names_corpus[i:i+batch_size_corpus]
        batch_emb = embed_model.encode(
            batch_texts,
            batch_size=batch_size_corpus,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        corpus_embeddings_list.append(batch_emb)
    S_corpus_np = np.concatenate(corpus_embeddings_list, axis=0).astype(np.float32)

    S_corpus_t = torch.from_numpy(S_corpus_np).to(device) 
    N, d = S_corpus_t.shape

    train_csv_path = "path to train dataset"
    print(f"Loading training data from: {train_csv_path}")
    train_df = pd.read_csv(train_csv_path, low_memory=False)

    titles = train_df['title'].fillna('').astype(str).tolist()
    abstracts = train_df['abstract'].fillna('').astype(str).tolist()
    subject_names_list = train_df['Subject Names'].fillna('').astype(str).tolist()
    n_rows = len(titles)
    print("Train set size:", n_rows)

    print("Building M embeddings (Title + Abstract in German)")
    combined_texts = [t + " " + a for (t, a) in zip(titles, abstracts)]
    M_np = embed_model.encode(
        combined_texts,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True
    )  
    
    print("Building matrix O embeddings (gold subject names) ...")
    O_list = []
    for subjects_str in subject_names_list:
        subjs = [x.strip() for x in subjects_str.split(';') if x.strip()]
        if len(subjs) == 0:
            O_list.append(np.zeros(d, dtype=np.float32))
            continue
        subjs_emb = embed_model.encode(
            subjs,
            batch_size=len(subjs),
            convert_to_numpy=True,
            show_progress_bar=False
        )
        O_list.append(subjs_emb.mean(axis=0))

    O_np = np.array(O_list, dtype=np.float32)

    M_t = torch.from_numpy(M_np).float().to(device)  
    O_t = torch.from_numpy(O_np).float().to(device)  
    print(f"M shape = {M_t.shape}, O shape = {O_t.shape}")

    transform = FocusedEmbTransform(d, hidden_dim=1024).to(device)

    margin = 0.2
    learning_rate = 1e-4
    weight_decay = 1e-5
    optimizer = optim.Adam(transform.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 30
    k_neg = 15 
    batch_size_train = 32

    from torch.utils.data import TensorDataset, DataLoader
    indices_all = torch.arange(n_rows)
    dataset = TensorDataset(indices_all)
    dataloader = DataLoader(dataset, batch_size=batch_size_train, shuffle=True)

    print("Starting Soft Retrieval Training (German) with Attention")
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            batch_indices = batch[0]
            loss_batch = 0.0

            for i_row in batch_indices:
                i_row = i_row.item()

                anchor = transform(M_t[i_row].unsqueeze(0))
                positive = O_t[i_row].unsqueeze(0)

                neg_indices = random.sample(range(N), k_neg)
                neg_embs = S_corpus_t[neg_indices] 

                dist_anchor_pos = distance_cosine(anchor, positive) 
                dist_anchor_neg = distance_cosine(anchor.expand(k_neg, -1), neg_embs) 

                margin_loss = margin + dist_anchor_pos - dist_anchor_neg  
                margin_loss = torch.clamp(margin_loss, min=0.0)
                loss_i = margin_loss.sum()
                loss_batch += loss_i

            loss_batch = loss_batch / len(batch_indices)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            epoch_loss += loss_batch.item()

        epoch_loss /= len(dataloader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{num_epochs}] LR={current_lr:.6f}, Avg Loss: {epoch_loss:.6f}")

    print("Training complete. Saving transform parameters to 'emb_transform.pt'")
    torch.save(transform.state_dict(), "emb_transform.pt")

if __name__ == "__main__":
    main()
