import os
import torch
import time
import pandas as pd 
from unsloth import FastLanguageModel
from unsloth import is_bf16_supported
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from pre_dataset import *

class FT_LLM:
    def __init__(self, model_name="unsloth/Qwen3-4B"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = 4096,   # Context length - can be longer, but uses more memory
            load_in_4bit = False,     # 4bit uses much less memory
            load_in_8bit = False,    # A bit more accurate, uses 2x memory
            full_finetuning = False, # We have full finetuning now!
            # token = "hf_...",      # use one if using gated models
        )
        
        self.model = FastLanguageModel.get_peft_model(
            model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 414,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )
        
        self.tokenizer = tokenizer
        
    def get_train_dataset_ready(self, df_input, task_name):
        dataset = Dataset.from_pandas(df_input)
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            fn_kwargs={
                "task_name": task_name,
                "mode": 'train',
            }
        )
        
        conversations = self.tokenizer.apply_chat_template(
            dataset["conversations"],
            tokenize = False,
            #add_generation_prompt = True, # Must add for generation
            #enable_thinking = True, # Enable thinking
        )
        
        data = pd.concat([pd.Series(conversations)])
        data.name = "text"
        combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
        combined_dataset = combined_dataset.shuffle(seed = 3407)
        return combined_dataset

    def train_with_lora(self, combined_dataset, output_dir):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = combined_dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 8,
                gradient_accumulation_steps = 4, # Use GA to mimic batch size!
                warmup_steps = 5,
                num_train_epochs = 3, # Set this for 1 full training run.
                #max_steps = 30,
                learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                report_to = "none", # Use this for WandB etc
                save_strategy = "epoch" 
            ),
        )

        trainer_stats = trainer.train()
        #if os.dir.exist(output_dir + "/lora_model"):
        if os.path.exists(output_dir + "/lora_model"):
            pass 
        else:
            os.makedirs(output_dir + "/lora_model", exist_ok=True)
        self.model.save_pretrained(output_dir + "/lora_model") # Local saving
        self.tokenizer.save_pretrained(output_dir + "/lora_model")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Model saved to", output_dir + "/lora_model")
    
        
class Inference_FT_LLM:
    def __init__(self, lora_path):
        print('Loading model from:', lora_path)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path, # YOUR MODEL YOU USED FOR TRAINING
         max_seq_length = 4096,
        load_in_4bit = False, # Set to False for 16bit LoRA
        )
        self.text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        return 
    
    def get_test_dataset_ready(self, df_input, task_name):
        dataset = Dataset.from_pandas(df_input)
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            fn_kwargs={
                "task_name": task_name,
                "mode": 'test'
            }
        )
        
        conversations = self.tokenizer.apply_chat_template(
            dataset["conversations"],
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            enable_thinking = True, # Enable thinking
        )
        
        data = pd.concat([pd.Series(conversations)])
        data.name = "text"
        combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
        #combined_dataset = combined_dataset.shuffle(seed = 3407)
        return combined_dataset
        
    def predict_instance(self, input_text, temperature=1.0, max_new_tokens=4096):
        inputs = self.tokenizer(
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        output_tokens = self.model.generate(**inputs,
                                     #streamer = self.text_streamer,      
                                     max_new_tokens = max_new_tokens,
                                     use_cache = True, 
                                     temperature = temperature,
                                     min_p = 0.1)
        prompt_length = inputs['input_ids'].shape[1]
        generated_text = self.tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)
        return generated_text
    
    def predict_batch(self, combined_dataset):
        outputs = []
        for i in range(len(combined_dataset)):
            input_text = combined_dataset[i]['text']
            inputs = self.tokenizer(
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to("cuda")
            output_tokens = self.model.generate(**inputs,
                                         streamer = self.text_streamer,      
                                         max_new_tokens = 4096,
                                         use_cache = True, 
                                         temperature = 1.0,
                                         min_p = 0.1)
            prompt_length = inputs['input_ids'].shape[1]
            generated_text = self.tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)
            #yield generated_text  
            outputs.append(generated_text)  
        return outputs
            
            