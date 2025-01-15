import os
import csv
import torch
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

GER_DATA_PATH = "/path to german database"
OUTPUT_FOLDER = "run1_de"
TOP_K = 50
SEARCH_BATCH_SIZE = 16
EMBED_BATCH_SIZE = 64
GERMAN_FAISS_INDEX_FILE = "faiss_index_german.index"
INPUT_FOLDER = "path to test/de"

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

def semantic_search(query, embed_model, faiss_index, top_k=50, batch_size=16):
    query_embeddings = embed_model.encode(
        [query],
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    distances, indices = faiss_index.search(query_embeddings, top_k)
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
    embed_model_de = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1', device=device)
    print("SentenceTransformer loaded for German.")

    if os.path.exists(GERMAN_FAISS_INDEX_FILE) and os.path.exists(GER_DATA_PATH):
        faiss_cpu_index = faiss.read_index(GERMAN_FAISS_INDEX_FILE)
        res = faiss.StandardGpuResources()
        faiss_gpu_index_de = faiss.index_cpu_to_gpu(res, 0, faiss_cpu_index)
        print("German FAISS index loaded.")
        codes_de, texts_de = read_data(GER_DATA_PATH)
    else:
        print("FAISS index or data file not found. Building the German FAISS index.")
        codes_de, texts_de = read_data(GER_DATA_PATH)
        faiss_gpu_index_de = create_vector_database(codes_de, texts_de, embed_model_de, batch_size=EMBED_BATCH_SIZE)
        faiss_cpu_index_de = faiss.index_gpu_to_cpu(faiss_gpu_index_de)
        faiss.write_index(faiss_cpu_index_de, GERMAN_FAISS_INDEX_FILE)
        print(f"German FAISS index saved to {GERMAN_FAISS_INDEX_FILE}.")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: {OUTPUT_FOLDER}")
    else:
        print(f"Output directory: {OUTPUT_FOLDER}")

    print(f"Processing German .jsonld files in folder: {INPUT_FOLDER}")
    jsonld_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".jsonld")]
    for file in tqdm(jsonld_files, desc="Processing German files", unit="file"):
        file_path = os.path.join(INPUT_FOLDER, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            jsonld_content = json.load(f)

        title, abstract = extract_title_abstract(jsonld_content)
        query = f"{title} {abstract}"

        indices, distances = semantic_search(query, embed_model_de, faiss_gpu_index_de, top_k=TOP_K, batch_size=SEARCH_BATCH_SIZE)
        retrieved_codes = [codes_de[idx] for idx in indices if idx < len(codes_de)]

        base_filename = os.path.splitext(file)[0]
        output_filename = base_filename + ".json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        save_retrieved_subjects(output_path, retrieved_codes)

    print("All German JSON-LD files processed successfully.")

if __name__ == "__main__":
    main()
