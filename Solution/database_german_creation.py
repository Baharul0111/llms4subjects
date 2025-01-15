import pandas as pd

data_path = '/fab3/btech/2022/baharul.islam22b/SEMEVAL2025/shared-task-datasets/GND/dataset/GND-Subjects-all.json'

df = pd.read_json(data_path)

def replace_commas(x):
    if isinstance(x, str):
        return x.replace(',', ';')
    return x

def list_to_semicolon_strings(x):
    if isinstance(x, list):
        return ';'.join(map(str, x))
    return x

df = df.fillna(' ')
df = df.applymap(list_to_semicolon_strings)
df = df.applymap(replace_commas)
print("Modified the dataset")
df.to_csv("Database_german.csv")
print("Dataset is saved as Database_german.csv")