import sys 
import json
import pandas as pd
from openai import OpenAI
from Dance import *

def merge_js_by_group_id(js_a_str, js_b_str, merge_key='category'):
    js_a = json_res_to_struct(js_a_str)
    js_b = json_res_to_struct(js_b_str)
    for i, term in enumerate(js_b):
        js_a[i][merge_key] = term[merge_key]
        
    js_a_str = json.dumps(js_a, indent=4)
    return js_a_str

def get_absis_res(orig_df, text_id):
    df_index = orig_df[orig_df.text_id == text_id]
    if df_index.shape[0] == 0:
        print(f'No text_id {text_id} found in the dataset')
        return -1
    input_doc = df_index.iloc[0]['input_doc']
    final_output = df_index.iloc[0]['final_output']
    cat_output = df_index.iloc[0]['category_data']
        
    absis_res = merge_js_by_group_id(final_output, cat_output, merge_key='category')
    #absis_res_js = json_res_to_struct(absis_res)
    return input_doc, absis_res

def meta_judge(input_doc, absis_gpt_4o, absis_gpt_35_turbo, absis_gpt_4o_mini, domain, model="o1-mini"):
    cat_list = RAG_categories_options(domain = domain)
    
    task_prompt = '''
    **Task:**  
    Perform Aspect-Based Sentiment Analysis (ABSA) on the given text and produce a **final, consolidated conclusion**. Your conclusion must be **based on the results from three separate LLM agents**, which will be provided. Specifically, you should:

    1. **Aspect Term:** Identify explicit substrings in the text that refer to aspects.

    2. **Aspect Category:** Assign each aspect term to one of the following predefined categories:  {}
    
    3. **Opinion Term:** Extract words or phrases (fewer than five words) that express the sentiment toward each aspect. Preserve their original form, including punctuation, spacing, and letter casing (e.g., 'love!!!!', 'gooood.', 'SOOO!!!!').

    4. **Sentiment Polarity:** Classify the sentiment of each aspect as positive, negative, or neutral.

    5. **Sentiment Intensity Score:** Assign an intensity score from -5 to 5 for each aspect-category-opinion-sentiment tuple. The scale should reflect the strength of the sentiment, where:  
    - Slight/moderate positive: 1 to 2  
    - Strong positive: 4 to 5  
    - Neutral or factual: 0  
    - Slight/moderate negative: -1 to -2  
    - Strong negative: -4 to -5  

    **Your final output must integrate and reconcile the differences from the results of the three LLM agents to produce one definitive ABSA result.** The three agents’ outputs, as well as the input text, will be provided to you. Use their analyses to guide your final decisions, ensuring the conclusion is grounded in their collective findings.

    **Note:** Please provide only the final output, preserving the same format as that used by the LLM agents.
    '''.format(cat_list)

    input_content = '''
    Input Document:
    {}

    Results from LLM Agent 1:
    {}

    Results from LLM Agent 2:
    {}

    Results from LLM Agent 3:
    {}
    '''.format(input_doc, absis_gpt_4o, absis_gpt_35_turbo, absis_gpt_4o_mini)

    meta_res = agent_analyse(input_content, task_prompt, model=model)
    return meta_res



