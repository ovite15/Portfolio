from transformers import WhisperForConditionalGeneration,WhisperProcessor ,WhisperTokenizer,WhisperFeatureExtractor
import evaluate
import torch
from datasets import load_from_disk , Audio ,DatasetDict
import os
from PIL import Image
import random
import evaluate
import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig,QuantoConfig
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import prepare_model_for_kbit_training 
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR





def main():
    # quantization_config = QuantoConfig(weights="int8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WhisperForConditionalGeneration.from_pretrained("whisper_L")
    model.to(device)

    ## finetune adapter lora ##
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    processor  = WhisperProcessor.from_pretrained("whisper_L", language="Thai", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("whisper_L", language="Thai", task="transcribe")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("whisper_L")

    model.generation_config.language = "Thai"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    ## load data that adjust to format in huggingface 
    ds = load_from_disk("prepare_data")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
    
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
    
            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
    
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    
            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
    
            batch["labels"] = labels
    
            return batch
            
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,)


    ## word error rate
    metric = evaluate.load("wer")


    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
    
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
    
        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
        return {"wer": wer}
    

    
    training_args = Seq2SeqTrainingArguments(
        output_dir="/output",  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        bf16=True,
        bf16_full_eval=True,
        num_train_epochs=1,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=2900,
        eval_steps=2900,
        logging_steps=25,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        deepspeed="ds_config.json", # deepspeed config
        push_to_hub=False, 
        remove_unused_columns=False
    )

    # save model and adapter 
    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control
        
    

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback]
    )

    model.config.use_cache = False
    trainer.train()

    trainer.save_model("/final_model")


if __name__ == "__main__":
    main()
