from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("./whisper_L")

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("./whisper_L", language="Thai", task="transcribe")

from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_name = "whisper_L"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name)


import os
import pandas as pd
from datasets import Dataset, DatasetDict, Audio

# Base directories for MP3 files
train_mp3_dir = 'mp3/mp3/train'
test_mp3_dir = 'mp3/mp3/test'
validation_mp3_dir = 'mp3/mp3/dev'

# Paths to your CSV files
train_csv_path = 'train.csv'
test_csv_path = 'test.csv'
validation_csv_path = 'dev.csv'

# Function to process dataframes
def process_dataframe_with_sentence(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    df['audio'] = df['audio'].apply(lambda x: os.path.join(base_dir, x))
    df = df[['audio', 'sentence']]  # Keep only the 'audio' and 'sentence' columns
    return df

def process_dataframe_without_sentence(csv_path, base_dir):
    df = pd.read_csv(csv_path)
    df['audio'] = df['audio'].apply(lambda x: os.path.join(base_dir, x))
    df = df[['audio']]  # Keep only the 'audio' column
    return df

# Process the CSV files
train_df = process_dataframe_with_sentence(train_csv_path, train_mp3_dir)
validation_df = process_dataframe_with_sentence(validation_csv_path, validation_mp3_dir)
test_df = process_dataframe_without_sentence(test_csv_path, test_mp3_dir)

# Convert the dataframes to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)

# Define the audio feature type
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
validation_dataset = validation_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'validation': validation_dataset
})

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("./whisper_L", language="Thai", task="transcribe")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    if "sentence" in batch:
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


for split in dataset_dict.keys():
    remove_columns = dataset_dict[split].column_names
    if split == 'test' and 'sentence' in remove_columns:
        remove_columns.remove('sentence')
    dataset_dict[split] = dataset_dict[split].map(prepare_dataset, remove_columns=remove_columns, num_proc=1)

# Save the processed dataset
dataset_dict.save_to_disk("./prepare_data")
