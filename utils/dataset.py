import torch
from torch.utils.data import Dataset
import pandas as pd
type2id = {'caution_and_advice':1,
 'displaced_people_and_evacuations':2,
 'infrastructure_and_utility_damage':3,
 'not_humanitarian':4,
 'other_relevant_information':5,
 'requests_or_urgent_needs':6,
 'rescue_volunteering_or_donation_effort':7,
 'sympathy_and_support':8}
 
class TextDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_len):
        self.texts, self.labels = self.data_process(tsv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def data_process(self,tsv_path):
        data = pd.read_csv(tsv_path, sep='\t')
        return data['tweet_text'].to_list(), data['class_label'].to_list()
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = type2id[self.labels[idx]]-1
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }