import torch
from transformers.trainer_utils import set_seed


class BooksDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.story = df.story
        self.targets = self.df.label
        self.max_len = max_len


    def __len__(self):
        return len(self.story)

    def __getitem__(self, index):
        story = str(self.story[index])
        inputs = self.tokenizer(
            story
            padding='longest', #this will actually be receiving things one by one
            max_length=self.max_len,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        sentence = self.tokenizer.decode(ids)
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
                
