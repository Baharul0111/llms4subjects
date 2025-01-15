import os
import csv
import torch
import json
import numpy as np
import faiss
from transformers import pipeline, T5Tokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ENG_DATA_PATH = "/path to english database"
OUTPUT_FOLDER = "run1_en"
TOP_K = 50
SEARCH_BATCH_SIZE = 16
EMBED_BATCH_SIZE = 64
ENGLISH_FAISS_INDEX_FILE = "faiss_index_english.index"
INPUT_FOLDER = "/path to test folder."

def read_data(csv_path):
    codes = []
    combined_texts = []
    required_cols = ['Code', 'Name', 'Alternate Name', 'Related Subjects', 'Definition', 'Classification Name']
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not all(col in row for col in required_cols):
                continue  
            code = row['Code'].strip()
            name = row['Name'].strip()
            alt_name = row['Alternate Name'].strip()
            class_name = row['Classification Name'].strip()
            related_subjects = row['Related Subjects'].strip().replace(';', ' ')
            definition = row['Definition'].strip()
            
            combined = f"{name} {alt_name} {related_subjects} {class_name} {definition}"
            codes.append(code)
            combined_texts.append(combined)
    
    return codes, combined_texts

def create_vector_database(codes, texts, embed_model, batch_size=64):
    embeddings_list = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding in batches", unit="batch"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embed_model.encode(
            batch_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        embeddings_list.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings_list).astype('float32')
    dim = embeddings.shape[1]
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(dim)
    faiss_gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    faiss_gpu_index.add(embeddings)
    print("FAISS index built. Total items in index:", faiss_gpu_index.ntotal)
    return faiss_gpu_index

def create_abstract_summary(summarizer, tokenizer, text, do_sample=False):
    if not text.strip():
        return text  
    word_count = len(text.split())
    if word_count < 40:
        return text
    elif word_count > 250:
        cropped_text = ' '.join(text.split()[:250])
        text = cropped_text 

    prefix = "summarize: "
    text = prefix + text

    input_tokens = tokenizer.encode(text, return_tensors='pt')
    input_length = input_tokens.shape[1]
    max_input_length = tokenizer.model_max_length
    if input_length > max_input_length:
        input_tokens = input_tokens[:, :max_input_length - 2]
        text = tokenizer.decode(input_tokens[0], skip_special_tokens=True)
        input_length = input_tokens.shape[1]

    adjusted_max_length = min(150, input_length)
    adjusted_min_length = min(40, adjusted_max_length - 10)
    adjusted_min_length = max(adjusted_min_length, 10)
    
    try:
        summary = summarizer(
            text,
            max_length=adjusted_max_length,
            min_length=adjusted_min_length,
            do_sample=do_sample,
            truncation=True
        )[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return text  

def semantic_search(query, embed_model, faiss_gpu_index, top_k=50, batch_size=16):
    query_embeddings = embed_model.encode(
        [query],
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    distances, indices = faiss_gpu_index.search(query_embeddings, top_k)
    return indices[0], distances[0]

def extract_title_abstract(jsonld_content):
    graph = jsonld_content.get("@graph", [])
    if not graph:
        return '', ''

    title = ''
    abstract = ''
    
    for item in graph:
        if 'title' in item and not title:
            t = item.get("title", "")
            if isinstance(t, str):
                title = t.strip()
            elif isinstance(t, list):
                title = ' '.join(str(x).strip() for x in t)
        if 'abstract' in item and not abstract:
            a = item.get("abstract", "")
            if isinstance(a, str):
                abstract = a.strip()
            elif isinstance(a, list):
                abstract = ' '.join(str(x).strip() for x in a)
        if title and abstract:
            break
    return title, abstract

def save_retrieved_subjects(output_path, subject_codes):
    output_data = {"dcterms_subject": subject_codes}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model_en = SentenceTransformer('all-mpnet-base-v2', device=device)
    print("SentenceTransformer loaded for English.")

    summarizer = pipeline("summarization", model="t5-large", device=0 if torch.cuda.is_available() else -1)
    tokenizer = T5Tokenizer.from_pretrained("t5-large")

    if os.path.exists(ENGLISH_FAISS_INDEX_FILE) and os.path.exists(ENG_DATA_PATH):
        faiss_cpu_index_en = faiss.read_index(ENGLISH_FAISS_INDEX_FILE)
        res_en = faiss.StandardGpuResources()
        faiss_gpu_index_en = faiss.index_cpu_to_gpu(res_en, 0, faiss_cpu_index_en)
        print("English FAISS index loaded.")
        codes_en, texts_en = read_data(ENG_DATA_PATH)
    else:
        print("FAISS index or data file not found. Building the English FAISS index.")
        codes_en, texts_en = read_data(ENG_DATA_PATH)
        faiss_gpu_index_en = create_vector_database(codes_en, texts_en, embed_model_en, batch_size=EMBED_BATCH_SIZE)
        faiss_cpu_index_en = faiss.index_gpu_to_cpu(faiss_gpu_index_en)
        faiss.write_index(faiss_cpu_index_en, ENGLISH_FAISS_INDEX_FILE)
        print(f"English FAISS index saved to {ENGLISH_FAISS_INDEX_FILE}.")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
    else:
        print(f"Output directory: {OUTPUT_FOLDER}")

    print(f"Processing English .jsonld files in folder: {INPUT_FOLDER}")
    jsonld_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jsonld")]
    for file in tqdm(jsonld_files, desc="Processing English files", unit="file"):
        file_path = os.path.join(INPUT_FOLDER, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            jsonld_content = json.load(f)

        title, abstract = extract_title_abstract(jsonld_content)
        query = f"{title} {abstract}"

        word_count = len(abstract.split())
        if word_count > 250:
            cropped_abstract = ' '.join(abstract.split()[:250])
            summarized_abstract = create_abstract_summary(summarizer, tokenizer, cropped_abstract)
            query = f"{title} {summarized_abstract}"
        elif word_count > 40:
            summarized_abstract = create_abstract_summary(summarizer, tokenizer, abstract)
            query = f"{title} {summarized_abstract}"

        indices, distances = semantic_search(query, embed_model_en, faiss_gpu_index_en, top_k=TOP_K, batch_size=SEARCH_BATCH_SIZE)
        
        retrieved_codes = [codes_en[idx] for idx in indices]

        base_filename = os.path.splitext(file)[0]
        output_filename = base_filename + ".json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        save_retrieved_subjects(output_path, retrieved_codes)

    print("All English JSON-LD files processed successfully.")

if __name__ == "__main__":
    main()
