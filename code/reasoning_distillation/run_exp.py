
from args_exp import init_exp_args

args = init_exp_args(['--gpu_id', '0', 
                      '--task_name', 'LLM_CoT_Reasoning', '--model_name', 'unsloth/Qwen3-4B', 
                      '--domain', 'all'])
print("Arguments: %s", args)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
import gc
import time
import torch
import pandas as pd
from pre_dataset import *
from ft_llm_lora import FT_LLM, Inference_FT_LLM

if __name__ == "__main__":
    root_path = 'your_root_path_here'  # Update with your actual root path
    df_train = pd.read_csv('your_path/train.csv')
    df_test = pd.read_csv('your_path/test.csv')
    task_name = args.task_name
    model_prefix = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
    if args.is_sft is False:
        print('Using original model for finetuning')
        FT_model = FT_LLM(args.model_name)
    else:
        # /fold_0/LLM_CoT_Reasoning/Qwen3-4B/lora_model
        print('Using LoRA model for finetuning')
        FT_model = FT_LLM(root_path + f'/{task_name}/{model_prefix}/lora_model')
    train_set = FT_model.get_train_dataset_ready(df_train, task_name = task_name)
    print('train_set size: ', len(train_set))
    print('train_set: ', train_set[0])
    output_dir = root_path + '/{}/{}'.format(task_name, model_prefix)

    FT_model.train_with_lora(train_set, output_dir = output_dir)
    print('Training finished!')
    
    # Load the trained model for inference
    lora_path = output_dir + '/lora_model'
    Inf_Qwen = Inference_FT_LLM(lora_path)
    test_dataset = Inf_Qwen.get_test_dataset_ready(df_test, task_name=task_name)

    res_list = [-1] * len(test_dataset)
    for index in range(len(test_dataset)):
        start_time = time.time()
        input_text = test_dataset[index]['text']
        res = Inf_Qwen.predict_instance(input_text)
        end_time = time.time()
        print(f'Index: {index}, Time: {end_time - start_time:.2f}s, \n, Input: {input_text}\n Response: {res}')
        res_list[index] = res
        df_test[model_prefix] = res_list
        df_test.to_csv(output_dir + '/pred_test_{}_{}.csv'.format(task_name, model_prefix), index=False)

