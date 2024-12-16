import sys 
import json
import pandas as pd
from openai import OpenAI
from Dance import *

def agent_template(input, prompt):
    res = agent_analyse(input, prompt)
    return res

aspect_thoughts_judge_prompt = '''
You are a Judgment Agent tasked with evaluating the results of an Aspect-Based Thought Grouping Agent. Your job is to assess each thought group based on the provided input document, ensuring that it follows the defined rules. Your evaluation should include the correctness of the grouping, the accuracy of the extracted aspect term, and the completeness of the sentences within each group.

## Steps to Evaluate
1. Review Input Document. Thoroughly read the input document to understand the context and the relationships between sentences and aspects.
2. Evaluate Each Group:
- Check if the group includes all relevant sentences focusing on the same aspect.
- Verify that the extracted aspect term is precise and aligns with the definition of aspect terms.
- Ensure the sentences in the group are cohesive and represent only one distinct aspect.
3. Provide Feedback: If a group is correct, mark it as "Correct"; If a group is incorrect, mark it as "Incorrect" and explain why. Reasons might include: missing relevant sentences, aspect term not precise, aspect term not extracted correctly, or sentences do not belong to the aspect.
4. Don't miss any group.
5. Output Format. The evaluation should be structured in the following JSON format:
[
    total_groups: <total number of groups>,
    {
        "group_id": <id>,
        "evaluation": "Correct" or "Incorrect",
        "reason": "<If incorrect, explain why, withine 15 words>"
    }
]
'''

def RAG_cat_promt(domain = 'restaurant'):
    category_list = RAG_categories_options(domain)

    cat_judge_prompt = '''
    You are a Judge Agent tasked with evaluating the category assignments. Your role is to verify whether each assigned category correctly matches the aspect-based thought group, based on the provided aspect term and sentences. If a category assignment is incorrect, you should provide the correct category in your feedback. 

    ## Available Categories: {}
    '''.format(category_list) + \
    '''
    ## Steps to Evaluate:
    1. Review Available Categories: Familiarize yourself with the list of possible categories provided.
    2. Evaluate Category Assignment: Ensure the assigned category is the most appropriate for the aspect term and sentences, considering the context and specific nuances mentioned.
    3. Provide Feedback:
        - Correct Assignment: If the category is appropriate, mark the evaluation as "Correct", no reason and correct category needed.
        - Incorrect Assignment: If the category is inappropriate, mark the evaluation as "Incorrect", provide a clear explanation (withine 15 words), and specify the correct category.
    4. Output Format. Your evaluation should be structured in the following JSON format:
    [
        {
            "group_id": <id>,
            "evaluation": "Correct" or "Incorrect",
            "reason": "<If incorrect, explain why, withine 15 words>",
            "correct_category": "<If incorrect, provide the correct category>"
        }
    ]
    '''
    return cat_judge_prompt

def merge_js_by_group_id(js_a_str, js_b_str, merge_key='category'):
    js_a = json_res_to_struct(js_a_str)
    js_b = json_res_to_struct(js_b_str)
    for i, term in enumerate(js_b):
        js_a[i][merge_key] = term[merge_key]
        
    js_a_str = json.dumps(js_a, indent=4)
    return js_a_str

opinion_judge_prompt = '''
You are a Judge Agent tasked with evaluating the opinion terms extracted by the Opinion Term Extraction Agent. Your role is to verify the correctness and completeness of the opinion terms for each aspect based thought group based on the provided sentences. If any extraction is incorrect or incomplete, you should provide the correct opinion terms in your feedback.

## Steps to Evaluate
1. Review the Sentences: Carefully read the sentences in each aspect based thought group to understand the context and identify expressions reflecting sentiment intensity.

2. Evaluate the Extracted Opinion Terms:
- Explicit Substrings: Ensure each opinion term is an exact substring from the sentences.
- Preserve Original Form: Check that punctuation, lengthening expressions (e.g., "soooo good"), and uppercase spelling are preserved.
- Length Requirement: Verify that each opinion term is less than 5 words.
- Completeness: Confirm that all significant opinion terms reflecting sentiment intensity have been extracted.

3. Provide Feedback:
Correct Extraction: If the opinion terms are appropriate and complete, mark the evaluation as "Correct". No reason and correct_opinion_terms needed.
Incorrect Extraction: If there are issues, mark the evaluation as "Incorrect", provide a clear explanation (withine 15 words), and supply the correct opinion terms.

4. Output Format:
Your evaluation should be structured in the following JSON format:
[
    {
        "group_id": <id>,
        "evaluation": "Correct" or "Incorrect",
        "reason": "<If incorrect, explain why, withine 15 words>",
        "correct_opinion_terms": ["<opinion_term_1>", "<opinion_term_2>", ...]
    }
]   
'''

sa_judge_prompt = '''
You are a Judge Agent tasked with evaluating the sentiment polarity assigned by the Sentiment Analysis Agent to each aspect-category-opinion tuple. Your role is to verify the correctness of the sentiment analysis based on the provided sentences and opinion terms.

## Steps to Evaluate

1. Review the Sentences and Opinion Terms: Carefully read the sentences and opinion terms for each thought group to understand the expressed sentiment toward the aspect term.

2. Determine the Correct Sentiment Polarity.

3. Provide Feedback:
   - Correct Assignment: If the sentiment polarity is correct, mark the evaluation as `"Correct"`. No reason and correct_sentiment_polarity needed.
   - Incorrect Assignment: If the polarity is incorrect, mark the evaluation as `"Incorrect"`, provide a clear explanation (within 15 words), and supply the correct sentiment polarity.

## Output Format

Your evaluation should be structured in the following JSON format:
```json
[
    {
        "group_id": <id>,
        "evaluation": "Correct" or "Incorrect",
        "reason": "<If incorrect, explain why, within 15 words>",
        "correct_sentiment_polarity": "<positive|negative|neutral>"
    }
]
```
'''


sia_judge_prompt = '''
You are a Judge Agent tasked with evaluating the sentiment intensity scores assigned by the Sentiment Analysis Agent to each aspect-category-opinion tuple. Your role is to verify the correctness of the sentiment intensity scores based on the provided sentences and opinion terms.

## Steps to Evaluate

1. Review the Sentences and Opinion Terms: Carefully read the sentences and opinion terms for each thought group to understand the expressed sentiment toward the aspect term.

2. Assess the Sentiment Intensity Score:
Sentiment intensity scores range from -5 to 5, reflecting sentiment strength. Slight/moderate terms score -1 to -2 (negative) or 1 to 2 (positive). Strong/emphatic terms (e.g., "very delicious," "AWESOME!!!") score 4-5, while neutral or factual terms score 0.
    |

4. Provide Feedback:
   - Correct Assignment: If the sentiment intensity score are appropriate, mark the evaluation as `"Correct"`. No reason and correct_sentiment_intensity_score needed.
   - Incorrect Assignment: If either the  intensity score is inappropriate, mark the evaluation as `"Incorrect"`, provide a clear explanation (withine 15 words), and supply the correct sentiment intensity score.

## Output Format

Your evaluation should be structured in the following JSON format:
```json
[
    {
        "group_id": <id>,
        "evaluation": "Correct" or "Incorrect",
        "reason": "<If incorrect, explain why, within 15 words>",
        "correct_sentiment_intensity_score": <number from -5 to 5>
    }
]
```
'''

def llm_judge(df_wait_judge, domain, index):
    text_id = df_wait_judge.iloc[index]['text_id']
    input_doc = df_wait_judge.iloc[index]['input_doc']
    thought_groups = df_wait_judge.iloc[index]['thought_groups']

    print('Judge for thought groups:')
    input_struct = '''inpt_doc: {} \n thought_groups: {}'''.format(input_doc, thought_groups)
    thoughts_res = agent_template(input_struct, aspect_thoughts_judge_prompt)
    print(thoughts_res)

    print('Judge for category: ')
    category_data = df_wait_judge.iloc[index]['category_data']
    input_struct = merge_js_by_group_id(thought_groups, category_data, merge_key='category')
    cat_judge_prompt = RAG_cat_promt(domain=domain)
    cat_res = agent_template(input_struct, cat_judge_prompt)
    print(cat_res)

    print('Judge for opinion terms: ')
    input_struct = df_wait_judge.iloc[index]['thought_groups_with_opinion_terms']
    opinion_res = agent_template(input_struct, opinion_judge_prompt)
    print(opinion_res)

    print('Judge for Sentiment analysis : ')
    input_struct = df_wait_judge.iloc[index]['final_output']
    sa_res = agent_template(input_struct, sa_judge_prompt)
    print(sa_res)
    
    print('Judge for Sentiment intensity score: ')
    input_struct = df_wait_judge.iloc[index]['final_output']
    sia_res = agent_template(input_struct, sia_judge_prompt)
    print(sia_res)
    return [text_id, thoughts_res, cat_res, opinion_res, sa_res, sia_res]

def LLM_judge_all(df_wait_judge, domain):
    data = []
    for index in range(df_wait_judge.shape[0]):
        try:
            judge_res = llm_judge(df_wait_judge, domain, index)
            
            data.append(judge_res)
        except:
            print('error in index: ', index)
            continue
    df_judge_res = pd.DataFrame(data, columns=['text_id', 'thoughts_res', 'cat_res', 'opinion_res', 'sa_res', 'sia_res'])
    return df_judge_res

def case_test():
    df_yelp_agents_50 = pd.read_csv('../data/your data path for Dance result')
    
    input_doc = df_yelp_agents_50.iloc[0]['input_doc']
    thought_groups = df_yelp_agents_50.iloc[0]['thought_groups']

    print('Judge for thought groups:')
    input_struct = '''inpt_doc: {} \n thought_groups: {}'''.format(input_doc, thought_groups)
    res = agent_template(input_struct, aspect_thoughts_judge_prompt)
    print(res)
    
    print('Judge for category: ')
    category_data = df_yelp_agents_50.iloc[0]['category_data']
    input_struct = merge_js_by_group_id(thought_groups, category_data, merge_key='category')
    cat_judge_prompt = RAG_cat_promt(domain = 'restaurant')
    res = agent_template(input_struct, cat_judge_prompt)
    print(res)
    
    print('Judge for opinion terms: ')
    input_struct = df_yelp_agents_50.iloc[0]['thought_groups_with_opinion_terms']
    res = agent_template(input_struct, opinion_judge_prompt)
    print(res)
    
    print('Judge for Sentiment analysis and intensity score: ')
    input_struct = df_yelp_agents_50.iloc[15]['final_output']
    res = agent_template(input_struct, sa_judge_prompt)
    print(res)
 
    
if __name__ == '__main__':
    case_test()
    
    
        
    