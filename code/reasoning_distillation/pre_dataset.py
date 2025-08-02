
import os
import random
import pandas as pd 
from datasets import Dataset


def RAG_categories_options(domain):
    if domain == 'restaurant':
        categories_options = ["location general", "food prices", "food quality", "food general", 
    "ambience general", "service general", "restaurant prices", 
    "drinks prices", "restaurant miscellaneous", "drinks quality", 
    "drinks style_options", "restaurant general", "food style_options"]
    elif domain == 'hotel':
        categories_options = ["location general", "food_drinks miscellaneous", "hotel comfort", "rooms comfort", "hotel cleanliness", "facilities miscellaneous", "service design_features", "facilities general", "food_drinks quality", "hotel prices", "hotel general", "rooms miscellaneous","service general", "facilities quality", "rooms general", "rooms design_features"]
    elif domain == 'laptop':
        categories_options = ["battery design_features", "battery general", "battery operation_performance", "battery quality", "company design_features", "company general", "company operation_performance", "company price", "company quality", "cpu design_features", "cpu general", "cpu operation_performance", "cpu price", "cpu quality", "display design_features", "display general", "display operation_performance", "display price", "display quality", "display usability", "fans&cooling design_features", "fans&cooling general", "fans&cooling operation_performance", "fans&cooling quality", "graphics design_features", "graphics general", "graphics operation_performance", "graphics usability", "hard_disc design_features", "hard_disc general", "hard_disc miscellaneous", "hard_disc operation_performance", "hard_disc price", "hard_disc quality", "hard_disc usability", "hardware design_features", "hardware general", "hardware operation_performance", "hardware price", "hardware quality", "hardware usability", "keyboard design_features", "keyboard general", "keyboard miscellaneous", "keyboard operation_performance", "keyboard portability", "keyboard price", "keyboard quality", "keyboard usability", "laptop connectivity", "laptop design_features", "laptop general", "laptop miscellaneous", "laptop operation_performance", "laptop portability", "laptop price", "laptop quality", "laptop usability", "memory design_features", "memory general", "memory operation_performance", "memory quality", "memory usability", "motherboard general", "motherboard operation_performance", "motherboard quality", "mouse design_features", "mouse general", "mouse usability", "multimedia_devices connectivity", "multimedia_devices design_features", "multimedia_devices general", "multimedia_devices operation_performance", "multimedia_devices price", "multimedia_devices quality", "multimedia_devices usability", "optical_drives design_features", "optical_drives general", "optical_drives operation_performance", "optical_drives usability", "os design_features", "os general", "os miscellaneous", "os operation_performance", "os price", "os quality", "os usability", "out_of_scope design_features", "out_of_scope general", "out_of_scope operation_performance", "out_of_scope usability", "ports connectivity", "ports design_features", "ports general", "ports operation_performance", "ports portability", "ports quality", "ports usability", "power_supply connectivity", "power_supply design_features", "power_supply general", "power_supply operation_performance", "power_supply quality", "shipping general", "shipping operation_performance", "shipping price", "shipping quality", "software design_features", "software general", "software operation_performance", "software portability", "software price", "software quality", "software usability", "support design_features", "support general", "support operation_performance", "support price", "support quality", "warranty general", "warranty quality"]
    
    return categories_options

def load_agent_prompt(domain='hotel'):
    cat_list = RAG_categories_options(domain)
    agent_prompt_dict = {}
    agent_prompt_dict['Aspect_Based_Thought_Group'] = """
    You are an Aspect-Based Thought Grouping Agent. Your task is to process the input document into structured aspect-based thought groups while ensuring accuracy, relevance, and completeness. 
    # Definition of Aspect Terms
    Aspect terms should capture:
    - Entities: Specific components of the provider's offerings (e.g., food, drinks, ambiance).
    - Services: Actions or interactions provided by the provider (e.g., service quality, speed, reservation process).
    - General Offerings: Broad categories related to the provider (e.g., "restaurant", "menu", "experience").
    - Specific Attributes: Aspect term should as specific as possible, e.g., "waitress" instead of "service", "sushi" instead of "food".

    # Follow these steps:
    1. Divide: Segment the input document into thought groups, where each group focuses on a single aspect term. 
    2. Split: If a sentence contains multiple aspects, divide it into smaller thought groups, each representing a distinct aspect.
    3. Collect: If multiple consecutive sentences represent the same aspect, combine them into one thought group.
    4. Extract: Identify and extract explicit aspect terms as exact words or phrases from the text (do not generate aspect terms). 
    5. Assign IDs: Assign a unique, auto-incrementing group_id to each thought group, starting from 1.
    
    Output the results in the json format:
    [
        {
            "group_id": <id>,
            "sentences": ["<sentence1>", "<sentence2>", ...],
            "aspect_term": "<aspect_term>"
        }
    ]
    """
    agent_prompt_dict['Cat'] = "You are a Category Assignment Agent. Assign the most suitable category to each aspect based thought group. Use one of these categories: {}.".format(cat_list) + \
"""
Input Format:
[
    {
        "group_id": <id>,
        "sentences": ["<sentence1>", "<sentence2>", ...],
        "aspect_term": "<aspect_term>"
    }
]

Output Json Format:
[
    {"group_id": <id>, "category": "<category>"}
]
"""

    agent_prompt_dict['Opinion'] = """
You are an Opinion Term Extraction Agent. Extract words or phrases reflecting sentiment intensity for each thought group. Opinion terms must be extracted as explicit substrings from the text, preserving punctuation, lengthening expressions, and uppercase spelling. Each opinion terms should be less than 5 words.
Output the results in the following format:
[
    {"group_id": <id>, "opinion_term": ["<opinion_term_1>", "<opinion_term_2>", ...]}
]
"""
    agent_prompt_dict['Sentiment'] = """
    You are a Sentiment Analysis Agent. For each aspect-category-opinion tuple, determine the sentiment polarity (positive, negative) and assign a sentiment intensity score. Sentiment intensity scores range from -5 to 5, reflecting sentiment strength. Slight/moderate terms score -1 to -2 (negative) or 1 to 2 (positive). Strong/emphatic terms (e.g., "very delicious," "AWESOME!!!") score 4-5, while neutral or factual terms score 0.
    
    Input Format:
    [
        {
            "group_id": <id>,
            "sentences": ["<sentence1>", "<sentence2>", ...],
            "aspect_term": "<aspect_term>",
            "opinion_term": "opinion_term": ["<opinion_term_1>", "<opinion_term_2>", ...]
        }
    ]
    
    Output the results in the following json format:
    [
        {
            "group_id": <id>,
            "sentiment_polarity": "<positive|negative|neutral>",
            "sentiment_intensity_score": "<-5 to 5>"
        }
    ]
    """
    
    agent_prompt_dict['LLM_CoT'] = f'''
Identify and analyze aspect-based sentiment with the given text by extracting detailed information on aspect categories, aspect terms, opinion terms, and assign a sentiment intensity score.

1. Aspect Term: Identify explicit substrings related to aspects in the text. 
2. Aspect Category: Assign each aspect term to one of the predefined categories: {cat_list}.

3. Opinion Term: Extract words or phrases that reflect the sentiment intensity while preserving their original form, including punctuation and lenghthening expressions and Uppercase spelling (e.g., 'love!!!!', 'gooood.', 'SOOO!!!!'). Opinion term should be less than 5 words.
4. Sentiment Polarity: Evaluate and classify the sentiment associated with each aspect into a positive, negative, or neutral category.
5. Sentiment Intensity Score: Assign a sentiment intensity score from -5 to 5 for each aspect-category-opinion-sentiment (ACOS) tuple. Sentiment intensity scores range from -5 to 5, reflecting sentiment strength. Slight/moderate terms score -1 to -2 (negative) or 1 to 2 (positive). Strong/emphatic terms (e.g., "very delicious," "AWESOME!!!") score 4-5, while neutral or factual terms score 0.

Steps:
Category Assignment: Match each aspect term with an appropriate category from the predefined list.
Aspect Term Identification: Scan the text for aspect terms; use specific sub-strings.
Opinion Term Extraction: Identify phrases representing the opinion, ensuring to maintain punctuation and expression forms.
Sentiment Analysis: Determine the sentiment polarity and assign a numerical intensity score from -5 to 5.
Output Format
Each result should be structured in List format, with the fields:

[[aspect_term, category, opinion_term, sentiment_polarity, sentiment_intensity_score], ...]

Notes:
1. Ensure aspect terms are explicit unless implied by pronouns.
2. All categories must align with the predefined list.
3. Evaluate opinions accurately in context, reflecting genuine sentiment scores.
4. Only return the final output in json format.
5. Lengthening expressions must be included in opinion terms.
'''

    agent_prompt_dict['CoT'] = f'''
Identify and analyze aspect-based sentiment with the given text by extracting aspect categories, aspect terms, opinion terms, and assign a sentiment intensity score.
Categories Options: {cat_list}.
'''

    agent_prompt_dict['Instance_COS'] = f'''
Identify and analyze aspect-based sentiment with the given text ans aspect_term by extracting aspect categories, opinion terms, and assign a sentiment intensity score.
Categories Options: {cat_list}.
'''
    return agent_prompt_dict

def get_task_label(input_label_struct_str, task_name='Aspect_Based_Thought_Group'):
    input_list = eval(input_label_struct_str)
    if task_name == 'Aspect_Based_Thought_Group':
        output_list = [
            {
                "group_id": item["group_id"],
                "sentences": item["sentences"],
                "aspect_term": item["aspect_term"]
            }
            for item in input_list
        ]
    elif task_name == 'Cat':
        output_list = [
            {
                "group_id": item["group_id"],
                "category": item["category"]
            }
            for item in input_list
        ]
    elif task_name == 'Opinion':
        output_list = [
            {
                "group_id": item["group_id"],
                "opinion_term": item["opinion_term"]
            }
            for item in input_list
        ]
    elif task_name == 'Opinion_input':
        output_list = [
            {
                "group_id": item["group_id"],
                "sentences": item["sentences"],
                "aspect_term": item["aspect_term"],
                "opinion_term": item["opinion_term"]
            }
            for item in input_list
        ]
    elif task_name == 'Sentiment':
        output_list = [
            {
                "group_id": item["group_id"],
                "sentiment_polarity": item["sentiment_polarity"],
                "sentiment_intensity_score": item["sentiment_intensity_score"]
            }
            for item in input_list
        ]
    elif task_name == 'Instance_COS':
        output_list = [
            {
                "group_id": item["group_id"],
                "aspect_term": item["aspect_term"],
                "category": item["category"],
                "opinion_term": item["opinion_term"],
                "sentiment_polarity": item["sentiment_polarity"],
                "sentiment_intensity_score": item["sentiment_intensity_score"]
            }
            for item in input_list
        ]
    return output_list
    
def get_sample(input, output, domain, task_name):
    input_doc = input
    agent_prompt_dict = load_agent_prompt(domain)
    think_chain = ''
    if task_name == 'Aspect_Based_Thought_Group':
        instruction = agent_prompt_dict[task_name]
        input_label_struct_str = output
        output = get_task_label(input_label_struct_str, task_name)
        return instruction, input_doc, output, think_chain
    elif task_name == 'Cat':
        instruction = agent_prompt_dict[task_name]
        input_label_struct_str = output
        input_doc = get_task_label(input_label_struct_str, 'Aspect_Based_Thought_Group')
        output = get_task_label(input_label_struct_str, task_name)
        return instruction, input_doc, output, think_chain
    elif task_name == 'Opinion':
        instruction = agent_prompt_dict[task_name]
        input_label_struct_str = output
        input_doc = get_task_label(input_label_struct_str, 'Aspect_Based_Thought_Group')
        output = get_task_label(input_label_struct_str, task_name)
        return instruction, input_doc, output, think_chain
    elif task_name == 'Sentiment':
        instruction = agent_prompt_dict[task_name]
        input_label_struct_str = output
        input_doc = get_task_label(input_label_struct_str, 'Opinion_input')
        output = get_task_label(input_label_struct_str, task_name)
        return instruction, input_doc, output, think_chain  
    elif task_name == 'LLM_CoT_Reasoning':
        instruction = agent_prompt_dict['LLM_CoT']
        input_label_struct_str = output
        aspect_based_thought_group_res = get_task_label(input_label_struct_str, 'Aspect_Based_Thought_Group')
        category_res = get_task_label(input_label_struct_str, 'Cat')
        opinion_res = get_task_label(input_label_struct_str, 'Opinion')
        sentiment_res = get_task_label(input_label_struct_str, 'Sentiment')
        
        think_chain = f"Okay, let's see.\n First I will do Aspect Based Thought Grouping, then I will do Category Assignment, then Opinion Term Extraction, and finally Sentiment Analysis.\n Let me think step by step.\n 1) Aspect Based Thought Grouping: \n {aspect_based_thought_group_res} \n 2) Category Assignment: \n {category_res} \n 3) Opinion Term Extraction: \n {opinion_res} \n 4) Sentiment Analysis: \n {sentiment_res} \n 4) Now I will merge all the results together: \n {input_label_struct_str} \n So I think that's the answer."
        output = input_label_struct_str
        return instruction, input_doc, output, think_chain
      
    else:
        print(f"Task {task_name} is not supported yet.")
        pass
    
    
def formatting_prompts_func(examples, task_name='LLM_CoT_Reasoning', mode='train'):
    domains = examples['domain']
    inputs = examples['input_doc']
    conversations = []

    if task_name in ['LLM_CoT_Reasoning']:
        outputs = examples['golden_label_struct']
        for input_doc, output, domain in zip(inputs, outputs, domains):
            instruction, input_doc, output, think_chain = get_sample(input_doc, output, domain, task_name)
            if mode == 'train':
                text = [
                {"role" : "user", "content" : f"Instruction:\n{instruction}\n Input:\n{input_doc}"},
                {"role" : "assistant", "content" : f'\n<think>\n{think_chain}\n</think>\n{output}'},
            ]   
            elif mode == 'test':
                text = [
                {"role" : "user", "content" : f"Instruction:\n{instruction}\n Input:\n{input_doc}"}
            ]
            conversations.append(text)
        return {"conversations": conversations}
    
    
    
 
 
