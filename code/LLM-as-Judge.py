import sys 
sys.path.append('../utils/')
import time
import pandas as pd 
import random 
import time 
from eval_helper import *
from eval_metrics import * 
from multi_agents_doc_absa import * 
from args_utils import *


from collections import Counter
from itertools import groupby

def normalize_entry(entry):
    """Normalize an entry by converting the text and values into a tuple."""
    return (entry[0], entry[1], entry[2].lower(), entry[3], int(entry[4]))

def calculate_average(entries):
    """Calculate the average of the 5th element for entries."""
    score_sum = sum(entry[4] for entry in entries)
    return score_sum // len(entries)

def union_based_aggregation(*lists):
    """Perform union-based aggregation to include every unique entry."""
    all_entries = []

    # Normalize and gather all entries from all lists
    for lst in lists:
        for entry in lst:
            all_entries.append(normalize_entry(entry))

    # Group entries by the first four elements
    grouped_entries = groupby(sorted(all_entries, key=lambda x: x[:4]), key=lambda x: x[:4])

    # Combine all unique entries, averaging the 5th element
    final_results = []
    for key, group in grouped_entries:
        group_list = list(group)
        average_score = calculate_average(group_list)
        final_results.append(list(key) + [average_score])

    return final_results

def majority_vote(*lists):
    """Perform majority vote to find the final result list."""
    all_entries = []

    # Normalize and gather all entries from all lists
    for lst in lists:
        for entry in lst:
            all_entries.append(normalize_entry(entry))

    # Group entries by the first four elements
    grouped_entries = groupby(sorted(all_entries, key=lambda x: x[:4]), key=lambda x: x[:4])

    # Select entries that appear in the majority of lists, averaging the 5th element
    final_results = []
    for key, group in grouped_entries:
        group_list = list(group)
        if len(group_list) >= len(lists) / 2:  # Majority threshold
            average_score = calculate_average(group_list)
            final_results.append(list(key) + [average_score])

    return final_results

def get_task_prompt(index):
    input_doc = df_label.iloc[index]['input_doc']

    absis_gpt_4o_str = df_label.iloc[index]['absis_gpt_4o']
    absis_gpt_4o_js = json_res_to_struct(absis_gpt_4o_str)
    absis_gpt_4o_sis_list = get_acos_sis_list(absis_gpt_4o_js)

    absis_gpt_35_turbo_str = df_label.iloc[index]['absis_gpt_35_turbo']
    absis_gpt_35_turbo_js = json_res_to_struct(absis_gpt_35_turbo_str)
    absis_gpt_35_turbo_sis_list = get_acos_sis_list(absis_gpt_35_turbo_js)

    absis_gpt_4o_mini_str = df_label.iloc[index]['absis_gpt_4o_mini']
    absis_gpt_4o_mini_js = json_res_to_struct(absis_gpt_4o_mini_str)
    absis_gpt_4o_mini_sis_list = get_acos_sis_list(absis_gpt_4o_mini_js)

    meta_res_str = df_label.iloc[index]['meta_res']
    meta_res_js = json_res_to_struct(meta_res_str)
    meta_res_sis_list = get_acos_sis_list(meta_res_js)

    # Find the final result list using both methods
    union_result_list = union_based_aggregation(absis_gpt_4o_sis_list, absis_gpt_35_turbo_sis_list, absis_gpt_4o_mini_sis_list)
    majority_result_list = majority_vote(absis_gpt_4o_sis_list, absis_gpt_35_turbo_sis_list, absis_gpt_4o_mini_sis_list)
    
    #Task_prompt =  Task_guidline + 'Input Document:\n' + input_doc + '\n\n' + 'Res 1:\n' + str(meta_res_sis_list) + '\n\n' + 'Res 2:\n' + str(union_result_list) + '\n\n' + 'Res 3:\n' + str(majority_result_list) + '\n\n'
    
    prompt = Task_guidline
    input_doc = 'Input Document:\n' + input_doc + '\n\n' + 'Res 1:\n' + str(meta_res_sis_list) + '\n\n' + 'Res 2:\n' + str(union_result_list) + '\n\n' + 'Res 3:\n' + str(majority_result_list) 

    return prompt, input_doc 





if __name__ == '__main__':
    args = init_args_agents()
    # Load the dataset
    if args.dataset_name == 'yelp':
        df_label_100 = pd.read_csv('../../dataset/absa_informal/meta/yelp_agents_absa_100_meta.csv')
        df_label_100_600 = pd.read_csv('../../dataset/absa_informal/meta/yelp_agents_absa_100_600_meta.csv')
        domain = 'restaurant'
    elif args.dataset_name == 'trip':
        df_label_100 = pd.read_csv('../../dataset/absa_informal/meta/trip_agents_absa_100_meta.csv')
        df_label_100_600 = pd.read_csv('../../dataset/absa_informal/meta/trip_agents_absa_100_600_meta.csv')
        domain = 'hotel'
    elif args.dataset_name == 'amz':
        df_label_100 = pd.read_csv('../../dataset/absa_informal/meta/amz_agents_absa_100_meta.csv')
        df_label_100_600 = pd.read_csv('../../dataset/absa_informal/meta/amz_agents_absa_100_600_meta.csv')
        domain = 'laptop'
        
    model = args.model_name
    print(args)
    df_label = pd.concat([df_label_100, df_label_100_600])
    # randomly sample 100 rows, with random seed 42
    df_label = df_label.sample(n=20, random_state=42)
    categories_options = RAG_categories_options(domain)
    Task_guidline= '''
Now you are an expert with ABSA task analysis and informal styles. You are required to evaluate the results of the three models:

# Task Guidline: Document level ABSA with sentiment intensity and informal styles

## Aspect terms should capture:
- Entities: Specific components of the provider's offerings (e.g., food, drinks, ambiance).
- Services: Actions or interactions provided by the provider (e.g., service quality, speed, reservation process).
- General Offerings: Broad categories related to the provider (e.g., "restaurant", "menu", "experience").
- Specific Attributes: Aspect term should as specific as possible, e.g., "waitress" instead of "service", "sushi" instead of "food".

## Category Assignment:
Assign the most suitable category to each aspect based thought group. Use one of these categories: {}.

## Opinion Term Extraction. 
Extract words or phrases reflecting sentiment intensity for each thought group. Opinion terms must be extracted as explicit substrings from the text, preserving punctuation, lengthening expressions, and uppercase spelling. Each opinion terms should be less than 5 words.
  
## Sentiment Analysis with Intensity Score.
Eetermine the sentiment polarity (positive, negative) and assign a sentiment intensity score. Sentiment intensity scores range from -5 to 5, reflecting sentiment strength. Slight/moderate terms score -1 to -2 (negative) or 1 to 2 (positive). Strong/emphatic terms (e.g., "very delicious," "AWESOME!!!") score 4-5, while neutral or factual terms score 0.

## result format: [(aspect_term, category, opinion_term, sentiment ploarity, sentiment_intensity_score), ....]
'''.format(categories_options) + \
'''
# Please annotate your high-level aggrement for the 3 Res as score from 1-5, 1 disagree, 5 is excellent. your answer should be a list [label-to-Res1, label-to-Res2, label-to-Res3], for example [5, 2, 1]

Note: Directly output the final answer without any reasoning or explanation.
Your answer should be a list [label-to-Res1, label-to-Res2, label-to-Res3], for example [5, 2, 1]. 
'''

    df_res = pd.DataFrame(columns=['index', 'domain', 'prompt', 'input_doc', 'res'])
    if args.mode == 'debug':
        run_num = 2
    elif args.mode == 'all':
        run_num = df_label.shape[0]
    for i in range(run_num):
        start_time = time.time()
        prompt, input_doc  = get_task_prompt(index=i)
        res = agent_analyse(input_doc, prompt, model=args.model_name)
        #res = random.randint(1, 5)
        end_time = time.time()
        # print time str
        print('time str: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        print('### index : {}\n### prompt : {}\n### input_doc : {}\n### res : {}\n### time_cost: {}'.format(i, prompt, input_doc, res, end_time - start_time))
        # add [i, prompot, input_doc, res] to df_res
        df_res.loc[i] = [i, domain, prompt, input_doc, res]
        df_res.to_csv('./res/judge_{}_{}.csv'.format(domain, model), index=False)
        time.sleep(2)
        
        


