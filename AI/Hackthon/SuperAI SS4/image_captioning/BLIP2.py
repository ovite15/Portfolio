from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration,BlipForConditionalGeneration
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
import evaluate
from pythainlp.translate import Translate
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")

model = Blip2ForConditionalGeneration.from_pretrained( "Salesforce/blip2-opt-6.7b-coco", torch_dtype=torch.float16)  # doctest: +IGNORE_RESULT
model.to(device)

print("load_model")

# if attention_mask is not None and (input_ids == pad_token_id).any():
#     logger.warn("display nice warning here....")


# Load the dataset
# dataset = load_dataset("Tuch/v0.4.1_codataset", split='train')
# print("load_dataset")
# dataset = dataset.shuffle(seed=42)
# print("load_shuffle")
# # # Perform a train-test split
# dataset = dataset.train_test_split(test_size=0.1)
# print("load_dataset")


# class ImageCaptioningDataset(Dataset):
#     def __init__(self, dataset, processor):
#         self.dataset = dataset
#         self.processor = processor

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         encoding = self.processor(images=item["image"], text=item["text"], padding="max_length",truncation=True, return_tensors="pt")
#         # remove batch dimension
#         encoding = {k:v.squeeze() for k,v in encoding.items()}
#         encoding['label'] = item['text']
#         return encoding

# print("1")
# train_dataset = ImageCaptioningDataset(dataset['train'], processor)
# print("ImageCaption")
# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=5)
# for batch in train_dataloader:
#     print(batch['pixel_values'].shape)
#     break

# output = model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],labels=batch["input_ids"])
# output.logits
# metric = evaluate.load("wer")
# print(metric)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# model.train()

# for epoch in range(10):
#   print("Epoch:", epoch)
#   for idx, batch in enumerate(train_dataloader):

#     print(idx, batch.keys(), batch)

#     input_ids = batch.pop("input_ids").to(device)
#     pixel_values = batch.pop("pixel_values").to(device)
#     attention_mask = batch.pop("attention_mask").to(device)

#     outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

#     loss = outputs.loss

#     print("Loss:", loss.item())

#     loss.backward()

#     optimizer.step()
#     optimizer.zero_grad()


enth = Translate('en', 'th')
df = pd.read_csv('/home/sivakorn/Image_Cap/BLIP2_1/sample_submission.csv')
en2th = Translate('en', 'th',use_gpu = True)

for i in tqdm(range(int(len(df)))):
  file_name = df.loc[i]['image_id']
  if file_name.split('/')[1] != 'food' and file_name.split('/')[1] != 'travel':
    file_path = f'/home/sivakorn/Image_Cap/BLIP2_1/{file_name}.jpg'
    image = Image.open(file_path)
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs,max_length=20,min_length=12,no_repeat_ngram_size=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    th_txt = en2th.translate(generated_text)
    print(generated_text, ':', th_txt)
    df.loc[i]['caption'] = th_txt
  else:
    file_path = f'/home/sivakorn/Image_Cap/BLIP2_1/test/{file_name}.jpg'
    image = Image.open(file_path)
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs,max_length=20,min_length=12, no_repeat_ngram_size=3)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    th_txt = en2th.translate(generated_text)
    print(generated_text, ':', th_txt)
    df.loc[i]['caption'] = th_txt


df.to_csv('submission.csv', index=False)
