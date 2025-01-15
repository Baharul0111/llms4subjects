import os
import cudf
import json
import glob

def retrieve_data_from_jsonld(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    title = ""
    abstract = ""
    dcterms_subject = ""
    graph = data.get('@graph', [])
    for item in graph:
        if 'title' in item:
            if isinstance(item['title'], str):
                title = item['title']
        if 'abstract' in item:
            if isinstance(item['abstract'], list):
                abstract = ' '.join([str(a) for a in item['abstract']])
            elif isinstance(item['abstract'], str):
                abstract = item['abstract']
        if 'dcterms:subject' in item:
            subjects = item['dcterms:subject']
            if isinstance(subjects, list):
                gnd_codes = []
                for subj in subjects:
                    subj.get('@id', '') if isinstance(subj.get('@id', ''), str) and subj.get('@id', '').startswith('gnd:') else ''
                for code in gnd_codes:
                    if code:
                        gnd_codes.append(code)        
                dcterms_subject = ';'.join(gnd_codes)
            elif isinstance(subjects, dict):
                gnd_code = subjects.get('@id', '')
                if isinstance(gnd_code, str) and gnd_code.startswith('gnd:'):
                    dcterms_subject = gnd_code
    return {
        'title': title , 'abstract': abstract , 'dcterms_subject': dcterms_subject 
    }

def process_jsonld_files(folder_path, output_csv):
    pattern = os.path.join(folder_path, '*.jsonld')
    files = glob.glob(pattern)
    retrieve_data = []
    for file in files:
        data = retrieve_data_from_jsonld(file)
        retrieve_data.append([
            data['title'] if isinstance(data['title'], str) else '',
            data['abstract'] if isinstance(data['abstract'], str) else '',
            data['dcterms_subject'] if isinstance(data['dcterms_subject'], str) else ''
        ])
    df = cudf.DataFrame(retrieve_data, columns=['title', 'abstract', 'dcterms_subject'])
    df = df.fillna('')
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv}")

if __name__ == "__main__":
    base_folder =  '/fab3/btech/2022/baharul.islam22b/SEMEVAL2025/shared-task-datasets/TIBKAT/all-subjects/data/'
    tasks = ['train', 'dev']
    main_folders = ['Article', 'Book', 'Conference', 'Report', 'Thesis']
    languages = ['en', 'de']
    
    
    for task in tasks:
        for main in main_folders:
            for lang in languages:
                folder_path = os.path.join(base_folder, task, main, lang)
                output_csv = f"{main}_{lang}_{task}.csv"
                process_jsonld_files(folder_path, output_csv)
