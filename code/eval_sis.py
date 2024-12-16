import json
from data_utils_rlf import * 

def get_absa_json_res(res_str):
    if 'json' in res_str:
        res_str = re.search(r'```json(.*?)```', res_str, re.DOTALL).group(1).strip()
    elif '```' in res_str:
        res_str = re.search(r'```(.*?)```', res_str, re.DOTALL).group(1).strip()

    try:
        res_str = res_str.replace('```json', '').replace('```', '').replace("'", '"')
        res_str = re.sub(r'(?<=[a-zA-Z])"(?=[a-zA-Z])', "'", res_str)
        json_res = json.loads(res_str)
        return json_res
    except:
        print('get_absa_json_res error!!!!')
        return -1
    
def get_struct_absa_res(df_tmp):
    data = []
    for i in range(df_tmp.shape[0]):
        row = df_tmp.iloc[i]
        absa_res = row['absa_results_js']
        for j in range(len(absa_res)):
            try:
                absa_res_j = absa_res[j]
                if 'opinion_term' in absa_res_j.keys():
                    aspect_category = absa_res_j['category']
                    aspect_term = absa_res_j['aspect_term']
                    opinion_term = absa_res_j['opinion_term']
                    sentiment_polarity = absa_res_j['sentiment_polarity']
                    sentiment_intensity_score = absa_res_j['sentiment_intensity_score']
                    data.append([i, row['text'], aspect_category, aspect_term, opinion_term, sentiment_polarity, sentiment_intensity_score])
                elif 'opinion term' in absa_res_j.keys():
                    aspect_category = absa_res_j['aspect category']
                    aspect_term = absa_res_j['aspect term']
                    opinion_term = absa_res_j['opinion term']
                    sentiment_polarity = absa_res_j['sentiment polarity']
                    sentiment_intensity_score = absa_res_j['sentiment intensity score']
                    data.append([i, row['text'], aspect_category, aspect_term, opinion_term, sentiment_polarity, sentiment_intensity_score])
            except:
                print('get_struct_absa_res error!!!!', j, absa_res_j)

    # save data to a dataframe
    df_absa = pd.DataFrame(data, columns=['sample_id', 'text', 'aspect_category', 'aspect_term', 'opinion_term', 'sentiment_polarity', 'sentiment_intensity_score']) 
    return df_absa

def opinion_contain_rlf(opinion_term):
    if isinstance(opinion_term, str):
        return contain_lengthening_word(opinion_term)
    else:
        return False  
    
def get_rlf_sis_res(df_res, top_n=200):
    if top_n != -1:
        df_tmp = df_res[:top_n]    
    df_tmp['absa_results_js'] = df_tmp['absa_results'].apply(get_absa_json_res)
    df_tmp = df_tmp[df_tmp['absa_results_js'] != -1]
    df_tmp_absa = get_struct_absa_res(df_tmp)
    df_tmp_absa['is_rlf'] = df_tmp_absa['opinion_term'].apply(opinion_contain_rlf)
    sis_res = df_tmp_absa.groupby(['is_rlf', 'sentiment_polarity'])['sentiment_intensity_score'].agg(['mean', 'std', 'count'])
    return sis_res, df_tmp_absa

def get_rlf_sis_res_simple(df_res, top_n=200):
    if top_n != -1:
        df_tmp = df_res[:top_n]    
    df_tmp['absa_results_js'] = df_tmp['absa_results'].apply(get_absa_json_res)
    df_tmp = df_tmp[df_tmp['absa_results_js'] != -1]
    df_tmp_absa = get_struct_absa_res(df_tmp)
    df_tmp_absa['is_rlf'] = df_tmp_absa['opinion_term'].apply(opinion_contain_rlf)
    df_tmp_absa['sentiment_intensity_score'] = df_tmp_absa['sentiment_intensity_score'].apply(lambda x: abs(float(x)))
    sis_res = df_tmp_absa.groupby(['is_rlf'])['sentiment_intensity_score'].agg(['mean', 'std', 'count'])
    return sis_res, df_tmp_absa

def get_struct_agents_absa_res(df_res_agents):
    data = []
    for index in range(df_res_agents.shape[0]):
        final_output = df_res_agents.iloc[index]['final_output']
        category_output = df_res_agents.iloc[index]['category_data']
        input_doc = df_res_agents.iloc[index]['input_doc']
        
        final_output_js = get_absa_json_res(final_output)
        category_output_js = get_absa_json_res(category_output)
        try:
            for i, cat_item in enumerate(category_output_js):
                final_output_js[i]['aspect category'] = cat_item['category']
            
            for output_item in final_output_js:
                aspect_category = output_item['aspect category']
                aspect_term = output_item['aspect_term']
                opinion_term = ','.join(output_item['opinion_term'])
                sentiment_polarity = output_item['sentiment_polarity']
                sentiment_intensity_score = output_item['sentiment_intensity_score']
                group_id = output_item['group_id']
                sentences = output_item['sentences']
                data.append([i, input_doc, group_id, sentences, aspect_category, aspect_term, opinion_term, sentiment_polarity, sentiment_intensity_score])  
        except:
            print('get_struct_agents_absa_res error!!!!', index, final_output_js, category_output_js)
            
    df_res_agents_struct = pd.DataFrame(data, columns=['index', 'input_doc', 'group_id', 'sentences', 'aspect_category', 'aspect_term', 'opinion_term', 'sentiment_polarity', 'sentiment_intensity_score'])
    return df_res_agents_struct

def get_agents_rlf_sis_res(df_res_agents_struct, rlf_col='opinion_term'):
    df_res_agents_struct['is_rlf'] = df_res_agents_struct[rlf_col].apply(opinion_contain_rlf)
    # only keep df_res_agents_struct when sentiment_intensity_score is from -5 to 5 string 
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    df_res_agents_struct['sentiment_intensity_score_flag'] = df_res_agents_struct['sentiment_intensity_score'].apply(lambda x: is_number(x))
    df_res_agents_struct = df_res_agents_struct[df_res_agents_struct['sentiment_intensity_score_flag'] == True]
    del df_res_agents_struct['sentiment_intensity_score_flag']
    df_res_agents_struct['sentiment_intensity_score'] = df_res_agents_struct['sentiment_intensity_score'].apply(lambda x: float(x))
    sis_res = df_res_agents_struct.groupby(['is_rlf', 'sentiment_polarity'])['sentiment_intensity_score'].agg(['mean', 'std', 'count'])
    return sis_res
    
    
def get_agents_rlf_sis_res_simple(df_res_agents_struct, rlf_col='opinion_term'):
    df_res_agents_struct['is_rlf'] = df_res_agents_struct[rlf_col].apply(opinion_contain_rlf)
    # only keep df_res_agents_struct when sentiment_intensity_score is from -5 to 5 string 
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    df_res_agents_struct['sentiment_intensity_score_flag'] = df_res_agents_struct['sentiment_intensity_score'].apply(lambda x: is_number(x))
    df_res_agents_struct = df_res_agents_struct[df_res_agents_struct['sentiment_intensity_score_flag'] == True]
    del df_res_agents_struct['sentiment_intensity_score_flag']
    df_res_agents_struct['sentiment_intensity_score'] = df_res_agents_struct['sentiment_intensity_score'].apply(lambda x: abs(float(x)))
    sis_res = df_res_agents_struct.groupby(['is_rlf'])['sentiment_intensity_score'].agg(['mean', 'std', 'count'])
    return sis_res





    
    
    
    

