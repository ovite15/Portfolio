import pandas as pd
import torchaudio
from tqdm import tqdm
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
test_df = pd.read_csv('test.csv')


# ตรวจสอบว่า CUDA พร้อมใช้งานหรือไม่
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โหลดตัวประมวลผลและโมเดล
processor = AutoProcessor.from_pretrained("./merged_model", language="Thai", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("./merged_model")

# Wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)

# Assuming test_df is already defined
# For example:
# test_df = pd.DataFrame({"audio": ["audio1.mp3", "audio2.mp3", ...]})

src_lang, tgt_lang = "tha", "tha"
batch_size = 16  # Process 16 audio files at once
sentences = []

def process_batch(audio_paths):
    batch_inputs = []
    for audio_path in audio_paths:
        # Load and resample audio file
        audio, orig_freq = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)  # ต้องเป็น 16 kHz
        audio = audio.squeeze().numpy()
        batch_inputs.append(audio)
    
    # Process batch
    inputs = processor(audio=batch_inputs, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    # Explicitly pad the input features to length 3000
    input_features = inputs["input_features"]
    padded_input_features = torch.nn.functional.pad(input_features, (0, 3000 - input_features.shape[-1]), mode='constant', value=0)
    
    if isinstance(model, torch.nn.DataParallel):
        generated_ids = model.module.generate(padded_input_features)
    else:
        generated_ids = model.generate(padded_input_features)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)

# ใช้ tqdm เพื่อแสดงแถบความคืบหน้า
for i in tqdm(range(0, len(test_df), batch_size), leave=True):
    batch_audio_paths = [f"/mp3/mp3/test/{audio}" for audio in test_df['audio'][i:i + batch_size]]
    batch_sentences = process_batch(batch_audio_paths)
    sentences.extend(batch_sentences)

# เพิ่มผลลัพธ์ลงใน DataFrame
test_df['sentence'] = sentences

test_df.to_csv("test.csv",index=False)
