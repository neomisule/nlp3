import sys
import os

# Add project root to sys.path if running as script
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import re
    import numpy as np
    # Monkey patch for numpy >= 1.24 and 2.0 compatibility with older keras_preprocessing
    if not hasattr(np, 'str'):
        np.str = np.str_
    if not hasattr(np, 'unicode_'):
        np.unicode_ = np.str_
    import pandas as pd
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from src.config import MAX_VOCAB_SIZE
    from src.utils import save_json, mkdirs
except ImportError as e:
    print(f"\nCRITICAL ERROR: Failed to import required modules.\nError: {e}\n")
    print("Please ensure you are running this script within the virtual environment.")
    print("Try running: venv\\Scripts\\python src/preprocess.py\n")
    sys.exit(1)

def clean_text(text: str) -> str:
    #lower it
    text = text.lower()
    #remove non-alphanumeric characters 
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    #remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_imdb_csv(path):
    df = pd.read_csv(path)
    #common column names
    if 'review' in df.columns and 'sentiment' in df.columns:
        texts = df['review'].astype(str).tolist()
        labels = (df['sentiment'] == 'positive').astype(int).tolist()
    elif 'text' in df.columns and 'label' in df.columns:
        texts = df['text'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    elif 'review' in df.columns and 'label' in df.columns:
        texts = df['review'].astype(str).tolist()
        labels = df['label'].astype(int).tolist()
    else:
        #fallback: assume first col = review, second col = label
        texts = df.iloc[:,0].astype(str).tolist()
        labels = df.iloc[:,1].astype(int).tolist()
    return texts, labels

def build_tokenizer(texts, max_words=MAX_VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer

def texts_to_padded_sequences(tokenizer, texts, seq_length):
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=seq_length, padding='post', truncating='post')
    return padded

def prepare_and_save(dataset_csv, out_dir="data", seq_lengths=[25,50,100], test_size=0.5, random_state=42):
    mkdirs(out_dir)
    texts, labels = load_imdb_csv(dataset_csv)
    texts = [clean_text(t) for t in texts]
    #50/50 split
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)
    tokenizer = build_tokenizer(X_train, max_words=MAX_VOCAB_SIZE)

    token_json = {
        "num_words": MAX_VOCAB_SIZE,
        "word_index_size": len(tokenizer.word_index)
    }
    save_json(os.path.join(out_dir, "tokenizer_meta.json"), token_json)

    for seq_len in seq_lengths:
        Xtr = texts_to_padded_sequences(tokenizer, X_train, seq_len)
        Xte = texts_to_padded_sequences(tokenizer, X_test, seq_len)
        np.save(os.path.join(out_dir, f"X_train_seq{seq_len}.npy"), Xtr)
        np.save(os.path.join(out_dir, f"X_test_seq{seq_len}.npy"), Xte)
        np.save(os.path.join(out_dir, f"y_train_seq{seq_len}.npy"), np.array(y_train))
        np.save(os.path.join(out_dir, f"y_test_seq{seq_len}.npy"), np.array(y_test))

    import pickle
    with open(os.path.join(out_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Saved preprocessed arrays to {out_dir}")
    return tokenizer

if __name__ == "__main__":
    # Default execution
    prepare_and_save("data/IMDB Dataset.csv")