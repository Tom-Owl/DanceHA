import argparse

def init_exp_args(custom_args=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--gpu_id",
        type=str,
        #required=True,
        default='0',
        choices=['0', '1', '2', '3'],
        help="GPU ID to use.",
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default='unsloth/Qwen3-4B',
        choices=['unsloth/Qwen3-4B', 'unsloth/Qwen3-14B'],
        help="Model name for finetuning",
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default='all',
        choices=['all', 'restaurant', 'hotel', 'laptop'],
        help="domain list",
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        default='auto',
        help="lora path",
    )
    
    if custom_args is not None:
        args = parser.parse_args(custom_args)
    else:
        args = parser.parse_args()
    #print("Arguments: %s", args)
    return args