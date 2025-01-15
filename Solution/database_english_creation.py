import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import os

def split_multi_values(text, delimiter=';'):
    for item in text.split(delimiter):
        if item:
            return [item.strip() for item in text.split(delimiter)]
        else:
            return [text.strip()]
        
def join_multi_values(translated_items, delimiter='; '):
    return delimiter.join(translated_items)

def translate_texts(texts, tokenizer, model, device, batch_size=16, report_interval=5000, delimiter=';'):
    translated_texts = []
    total_rows = len(texts)
    split_texts = []
    for text in texts:
        split_texts.append(split_multi_values(text, delimiter))

    flat_texts = []
    index_map = []
    for idx, items in enumerate(split_texts):
        for item in items:
            if item: 
                flat_texts.append(item)
                index_map.append(idx)
            else:
                translated_texts.append('')

    translated_results = [''] * len(texts) 
    if flat_texts:
        translated_flat = []
        for i in tqdm(range(0, len(flat_texts), batch_size), desc="Translating"):
            batch = flat_texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                translated = model.generate(**inputs)
            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            translated_flat.extend(decoded)


            if (i + batch_size) % report_interval < batch_size:
                print(f'Translated {min(i + batch_size, len(flat_texts))} / {len(flat_texts)} items.')
        
        from collections import defaultdict
        translations_dict = defaultdict(list)
        for idx, translation in zip(index_map, translated_flat):
            translations_dict[idx].append(translation)
            
        for idx, items in enumerate(split_texts):
            if any(item for item in items):  
                translated_items = translations_dict[idx]
                if len(translated_items) != len([item for item in items if item]):
                    raise ValueError(f"Mismatch in number of translations for row {idx}.")
                translated_texts.append('; '.join(translated_items))
               
            else:
                translated_texts.append('')

    return translated_texts

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    input_csv_path = 'Database_german.csv'   
    output_csv_path = 'Database_english.csv'  
    columns_to_translate = [
        'Classification Name',
        'Name',
        'Alternate Name',
        'Related Subjects',
        'Source',
        'Definition',
        'Source URL'
    ]

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"The input CSV file '{input_csv_path}' does not exist.")

    df = pd.read_csv(input_csv_path, dtype=str) 
    missing = []
    for col in columns_to_translate + ['Code', 'Classification Number']:
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"The following required columns are missing in the input CSV: {missing}")

    df[columns_to_translate] = df[columns_to_translate].fillna('')

    model_name = 'Helsinki-NLP/opus-mt-de-en'
    print(f'Loading model and tokenizer: {model_name}')
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)

    translated_columns = {}

    for column in columns_to_translate:
        print(f'\nTranslating column: {column}')
        texts = df[column].tolist()
        translated = translate_texts(
            texts,
            tokenizer,
            model,
            device,
            batch_size=16,
            report_interval=5000,
            delimiter=';'
        )
        translated_columns[f'{column}_EN'] = translated

    for col, translated in translated_columns.items():
        df[col] = translated

    df.to_csv(output_csv_path, index=False)
    print(f'\nTranslation completed. Translated file saved to {output_csv_path}')

if __name__ == "__main__":
    main()
