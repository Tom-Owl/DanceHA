import json
from openai import OpenAI
import re
key_base_pth = 'your_key'
key_file_path = key_base_pth + 'open_ai_key.json'
with open(key_file_path) as f:
    load_data = json.load(f)
    
client = OpenAI(
        api_key=load_data['OPENAI_API_KEY'],
    )

def agent_analyse(input_doc, prompt, model='gpt-4o'):
    # "gpt-3.5-turbo",  "gpt-4o", "gpt-4o-mini"
    if model in ["gpt-3.5-turbo",  "gpt-4o", "gpt-4o-mini", "gpt-4o-2024-11-20", "gpt-4o-2024-05-13"]:
        #print('model_A:', model)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": input_doc
                    }
                ]
                }
            ],
            response_format={
                "type": "text"
            },
            temperature=1,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    elif model in ["o1-preview", "o1-mini"]:
        #print('model_B:', model)
        response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": input_doc
                }
            ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    else:
        print('model not supported')
        return -1
    try:
        res = response.choices[0].message.content
    except:
        response
    return res

def agent_aspect_based_thought_grouping(input_doc, model='gpt-4o'):
    """
    Agent 1: Aspect-Based Thought Grouping
    Segments text into thought groups and extracts aspect terms.
    """
    prompt = """
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
    return agent_analyse(input_doc, prompt, model)

def agent_aspect_with_split_cmp(input_doc, model='gpt-4o'):
    
    prompt = """
    You need to split the sentence into shorter sentences such that each short sentence contains one aspect term. When splitting, sentences connected by conjunctions must be divided into individual sentences along with their conjunctions. This process must specify the subject in every sentence. This process must retain the existing spellings exactly as in the original sentence. This process must also retain the existing spacings exactly as in the original sentence. If the sentence is too short to split or does not need to be split, use the original sentence as is. No numbering, line breaks, or explanations are needed.
    
    # Definition of Aspect Terms
    The ‘aspect’ refers to a specific feature, attribute, or aspect of a product or service that a user may express an opinion about. The aspect term might be ‘null’ for an implicit aspect.
    
    Output the results in the json format:
    [
        {
            "group_id": <id> (auto-incrementing group_id to each thought group, starting from 1),
            "sentences": "<sentence1>",
            "aspect_term": "<aspect_term>"
        }
    ]
    """
    return agent_analyse(input_doc, prompt, model)

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
    

def agent_category_assignment(thought_groups, domain='restaurant', model='gpt-4o'):
    """
    Agent 2: Category Assignment
    Assigns a predefined category to each aspect term.
    """
    categories_options = RAG_categories_options(domain)
    
    prompt = "You are a Category Assignment Agent. Assign the most suitable category to each aspect based thought group. Use one of these categories: {}.".format(categories_options) + \
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
    return agent_analyse(thought_groups, prompt, model)


def agent_opinion_term_extraction(thought_groups, model='gpt-4o'):
    """
    Agent 3: Opinion Term Extraction
    Extracts sentiment-bearing opinion terms from each thought group. Each opinion terms should be less than 5 words.
    """
    prompt = """
    You are an Opinion Term Extraction Agent. Extract words or phrases reflecting sentiment intensity for each thought group. Opinion terms must be extracted as explicit substrings from the text, preserving punctuation, lengthening expressions, and uppercase spelling. Each opinion terms should be less than 5 words.
    Output the results in the following format:
    [
        {"group_id": <id>, "opinion_term": ["<opinion_term_1>", "<opinion_term_2>", ...]}
    ]
    """
    return agent_analyse(thought_groups, prompt, model)

def json_res_to_struct(res_str):
    if 'json' in res_str:
        res_str = re.search(r'```json(.*?)```', res_str, re.DOTALL).group(1).strip()
    elif '```' in res_str:
        res_str = re.search(r'```(.*?)```', res_str, re.DOTALL).group(1).strip()

    try:
        res_str = res_str.replace('```json', '').replace('```', '')
        json_res = json.loads(res_str)
        return json_res
    except:
        print('error!!!!', res_str)
        return -1
    
def merge_opinion_term_to_thought_groups(thought_groups, opinion_terms):
    """
    Merges opinion terms with aspect-based thought groups.
    """
    thought_groups_js = json_res_to_struct(thought_groups)
    opinion_terms_js = json_res_to_struct(opinion_terms)
    for i, opinion_term in enumerate(opinion_terms_js):
        thought_groups_js[i]['opinion_term'] = opinion_term['opinion_term']
    return thought_groups_js

def agent_sentiment_analysis(thought_groups_with_opinion_terms, model='gpt-4o'):
    """
    Agent 4: Sentiment Analysis
    Determines sentiment polarity and intensity for each aspect-category-opinion tuple.
    """
    prompt = """
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
    return agent_analyse(thought_groups_with_opinion_terms, prompt, model) 

def merge_SA_to_thought_groups_with_opinion_terms(thought_groups_with_opinion_terms, sa_data):
    """
    Merges opinion terms with aspect-based thought groups.
    """
    thought_groups_with_opinion_terms_js = json_res_to_struct(thought_groups_with_opinion_terms)
    sa_data_js = json_res_to_struct(sa_data)
    for i, sa_item in enumerate(sa_data_js):
        thought_groups_with_opinion_terms_js[i]['sentiment_polarity'] = sa_item['sentiment_polarity']
        thought_groups_with_opinion_terms_js[i]['sentiment_intensity_score'] = sa_item['sentiment_intensity_score']
    return thought_groups_with_opinion_terms_js

def merge_js_by_group_id(js_a_str, js_b_str, merge_key='category'):
    js_a = json_res_to_struct(js_a_str)
    js_b = json_res_to_struct(js_b_str)
    for i, term in enumerate(js_b):
        js_a[i][merge_key] = term[merge_key]
        
    js_a_str = json.dumps(js_a, indent=4)
    return js_a_str

def multi_agent_document_level_absa(input_doc, domain='restaurant', model='gpt-4o', debug=False):
    # Step 1: Aspect-Based Thought Grouping
    if debug:
        print('input_doc:', input_doc)
    thought_groups = agent_aspect_based_thought_grouping(input_doc, model)
    if debug:
        print('thought_groups:', thought_groups)
    # Step 2 & 3: Parallel Processing for Category Assignment and Opinion Term Extraction
    category_data = agent_category_assignment(thought_groups, domain, model)
    if debug:
        print('category_data:', category_data)
    opinion_data = agent_opinion_term_extraction(thought_groups, model)
    if debug:
        print('opinion_data:', opinion_data)
    thought_groups_with_opinion_terms = merge_opinion_term_to_thought_groups(thought_groups, opinion_data)
    if debug:
        print('thought_groups_with_opinion_terms:', thought_groups_with_opinion_terms)
    thought_groups_with_opinion_terms_string = json.dumps(thought_groups_with_opinion_terms, indent=4)
    # Step 4: Sentiment Analysis
    sa_data = agent_sentiment_analysis(thought_groups_with_opinion_terms_string, model)
    if debug:
        print('sa_data:', sa_data)
    final_output = merge_SA_to_thought_groups_with_opinion_terms(thought_groups_with_opinion_terms_string, sa_data)
    
    
    final_output_string = json.dumps(final_output, indent=4)
    if debug:
        print('final_output:', final_output_string)

    final_output_string = merge_js_by_group_id(final_output_string, category_data)
    # rewrite the hist into dict format
    hist_dict = {
        'input_doc': input_doc,
        'thought_groups': thought_groups,
        'category_data': category_data,
        'opinion_data': opinion_data,
        'thought_groups_with_opinion_terms': thought_groups_with_opinion_terms_string,
        'sa_data': sa_data,
        'final_output': final_output_string
    }
    
    return final_output_string, hist_dict

def fast_test():
    input_doc = '''Oh. My. God. Cookie dough by the scoop?! Dough Nation, where have you been all my life?! Great concept and great execution. Totally reasonable prices, in my opinion, considering the delicious decadence you're paying for. I got an ice cream sandwich--you choose your cookies from several delightful options, and the ice cream filling was equivalent to at least a couple of scoops... All for 6 bucks! Sooo worth it'''

    for model in ["gpt-3.5-turbo",  "gpt-4o", "gpt-4o-mini", "o1-mini"]:
        print('Now using model:', model)
        multi_agent_document_level_absa(input_doc, domain='restaurant', model=model, debug=True)
        
def cmp_dance_divider(input_doc, domain='restaurant', model='gpt-4o', debug=False):
    # Step 1: Aspect-Based Thought Grouping
    if debug:
        print('input_doc:', input_doc)
    thought_groups = agent_aspect_with_split_cmp(input_doc, model)
    
    # rewrite the hist into dict format
    hist_dict = {
        'input_doc': input_doc,
        'thought_groups': thought_groups
    }
    
    return thought_groups, hist_dict

############
## Ablation Study w/o DC, w/o Teamwork
############
def w_o_DC(input_doc, domain, model='gpt-4o'):
    categories_options = RAG_categories_options(domain)
    
    prompt = """
    ## Task1: Aspect Extraction
    
    ### Definition of Aspect Terms
    Aspect terms should capture:
    - Entities: Specific components of the provider's offerings (e.g., food, drinks, ambiance).
    - Services: Actions or interactions provided by the provider (e.g., service quality, speed, reservation process).
    - General Offerings: Broad categories related to the provider (e.g., "restaurant", "menu", "experience").
    - Specific Attributes: Aspect term should as specific as possible, e.g., "waitress" instead of "service", "sushi" instead of "food".

    ## Task2: Category Assignment. 
    Assign the most suitable category to each aspect based thought group. Use one of these categories: {}.

    ## Task3: Opinion Term Extraction. 
    Extract words or phrases reflecting sentiment intensity for each thought group. Opinion terms must be extracted as explicit substrings from the text, preserving punctuation, lengthening expressions, and uppercase spelling. Each opinion terms should be less than 5 words.

    ## Task4: Sentiment Analysis with Intensity Score. 
    Eetermine the sentiment polarity (positive, negative) and assign a sentiment intensity score. Sentiment intensity scores range from -5 to 5, reflecting sentiment strength. Slight/moderate terms score -1 to -2 (negative) or 1 to 2 (positive). Strong/emphatic terms (e.g., "very delicious," "AWESOME!!!") score 4-5, while neutral or factual terms score 0.
    
    """.format(categories_options) + \
    """
    ### Output format:
    Output the results in the following json format:
        [
            {
                "group_id": <id>,
                "aspect_term": "<aspect_term>",
                "category": "<category>",
                "opinion_term": "opinion_term": ["<opinion_term_1>", "<opinion_term_2>", ...]
                "sentiment_polarity": "<positive|negative|neutral>",
                "sentiment_intensity_score": "<-5 to 5>"
            }
            ....
        ]
        
    Note: Finish all the task and directly output the final results.
    """
    
    #print('Now using promt:', prompt)
    return agent_analyse(input_doc, prompt, model)

def w_o_Teamwork(thought_group, domain, model='gpt-4o'):
    categories_options = RAG_categories_options(domain)
    
    prompt = """
    For each group, finish the following tasks:

    ## Task1: Category Assignment. 
    Assign the most suitable category to each aspect based thought group. Use one of these categories: {}.

    ## Task2: Opinion Term Extraction. 
    Extract words or phrases reflecting sentiment intensity for each thought group. Opinion terms must be extracted as explicit substrings from the text, preserving punctuation, lengthening expressions, and uppercase spelling. Each opinion terms should be less than 5 words.

    ## Task3: Sentiment Analysis with Intensity Score. 
    Eetermine the sentiment polarity (positive, negative) and assign a sentiment intensity score. Sentiment intensity scores range from -5 to 5, reflecting sentiment strength. Slight/moderate terms score -1 to -2 (negative) or 1 to 2 (positive). Strong/emphatic terms (e.g., "very delicious," "AWESOME!!!") score 4-5, while neutral or factual terms score 0.
    """.format(categories_options) + \
    """
    ### Input format:
    [
        {
            "group_id": <id>,
            "sentences": ["<sentence1>", "<sentence2>", ...],
            "aspect_term": "<aspect_term>"
        }
        ...
    ]
    
    ### Output format:
    Output the results in the following json format:
        [
            {
                "group_id": <id>,
                "sentences": ["<sentence1>", "<sentence2>", ...],
                "aspect_term": "<aspect_term>",
                "category": "<category>",
                "opinion_term": "opinion_term": ["<opinion_term_1>", "<opinion_term_2>", ...]
                "sentiment_polarity": "<positive|negative|neutral>",
                "sentiment_intensity_score": "<-5 to 5>"
            }
            ....
        ]
        
    Note: Finish all the task and directly output the final results.
    """
    
    #print('Now using promt:', prompt)
    return agent_analyse(thought_group, prompt, model)
    

if __name__ == '__main__':
    input_doc = '''Oh. My. God. Cookie dough by the scoop?! Dough Nation, where have you been all my life?! Great concept and great execution. Totally reasonable prices, in my opinion, considering the delicious decadence you're paying for. I got an ice cream sandwich--you choose your cookies from several delightful options, and the ice cream filling was equivalent to at least a couple of scoops... All for 6 bucks! Sooo worth it'''

    for model in ["gpt-3.5-turbo",  "gpt-4o", "gpt-4o-mini", "o1-mini"]:
        print('Now using model:', model)
        multi_agent_document_level_absa(input_doc, domain='restaurant', model=model, debug=True)
    
    
    