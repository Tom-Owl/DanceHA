import sys 
import pandas as pd 
from eval_metrics import *
from MA import *

def get_acos_list(js_res):
    acos_list = []
    try:
        for item in js_res:
            if 'opinion_term' in item.keys():
                for opinion in item['opinion_term']:
                    acos_list.append([item['aspect_term'], item['category'], opinion, item['sentiment_polarity']])
            elif 'opinion term' in item.keys():
                acos_list.append([item['aspect term'], item['aspect category'], item['opinion term'], item['sentiment polarity']])
            else:
                print('No opinion term found in the result')
        return acos_list
    except Exception as e:
        print(e)
        return acos_list
    
def get_acos_sis_list(js_res):
    acos_list = []
    try:
        for item in js_res:
            if 'opinion_term' in item.keys():
                for opinion in item['opinion_term']:
                    acos_list.append([item['aspect_term'], item['category'], opinion, item['sentiment_polarity'], item['sentiment_intensity_score']])
            elif 'opinion term' in item.keys():
                acos_list.append([item['aspect term'], item['aspect category'], item['opinion term'], item['sentiment polarity'], item['sentiment intensity score']])
            else:
                print('No opinion term found in the result')
        return acos_list
    except Exception as e:
        print(e)
        return acos_list
    
def eval_instance(df_res, index, pred_col, label_col='meta_res', is_sis=False):
    if is_sis is False:
        meta_res_str = df_res.iloc[index][label_col]
        meta_res = json_res_to_struct(meta_res_str)
        meta_acos = get_acos_list(meta_res)

        gpt_4o_res_str = df_res.iloc[index][pred_col]
        gpt_4o_res = json_res_to_struct(gpt_4o_res_str)
        gpt_4o_acos = get_acos_list(gpt_4o_res)

        if len(gpt_4o_acos) == 0 or len(meta_acos) == 0:
            print('error index: ', index)
            return 0, 0, 0
        eval_tool = F1ScoreSeq2Seq()
        eval_tool.eval([gpt_4o_acos], [meta_acos])
        return eval_tool.tp, eval_tool.fp, eval_tool.fn
    
    elif is_sis is True:
        meta_res_str = df_res.iloc[index][label_col]
        meta_res = json_res_to_struct(meta_res_str)
        meta_acos = get_acos_list(meta_res)
        meta_acos_sis = get_acos_sis_list(meta_res)

        gpt_4o_res_str = df_res.iloc[index][pred_col]
        gpt_4o_res = json_res_to_struct(gpt_4o_res_str)
        gpt_4o_acos = get_acos_list(gpt_4o_res)
        gpt_4o_acos_sis = get_acos_sis_list(gpt_4o_res)

        if len(gpt_4o_acos) == 0 or len(meta_acos) == 0:
            print('error index: ', index)
            return 0, 0, 0
        eval_tool = F1ScoreSeq2Seq()
        mae_list = eval_tool.eval_mae([gpt_4o_acos], [meta_acos], [gpt_4o_acos_sis], [meta_acos_sis])
        return mae_list
    else:
        print('is_sis should be either True or False')

def eval_res(df_res, pred_col='absis_gpt_4o', label_col='meta_res', is_sis=False):
    if is_sis is False:
        data = []
        for i in range(df_res.shape[0]):
            tp, fp, fn = eval_instance(df_res, index=i, pred_col=pred_col, label_col=label_col)
            data.append([tp, fp, fn])
            
        df_tmp = pd.DataFrame(data, columns=['tp', 'fp', 'fn'])
        fp = df_tmp['fp'].sum()
        tp = df_tmp['tp'].sum()
        fn = df_tmp['fn'].sum()
        precision = tp / (tp + fp)  
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        acc = tp / (tp + fp + fn)
        return df_tmp, precision, recall, f1, acc, tp, fp, fn
    elif is_sis is True:
        data = []
        for i in range(df_res.shape[0]):
            mae_list = eval_instance(df_res, index=i, pred_col=pred_col, label_col=label_col, is_sis=True)
            data.append(mae_list)
            
        # flatted the list and get average
        mae_list = [item for sublist in data for item in sublist]
        mae = sum(mae_list) / len(mae_list)
        return data, mae
        


def eval_instance_vanilla(df_res, df_vanilla, index, pred_col='absa_results', label_col='meta_res', is_sis=False):
    if is_sis is False:
        text_id = df_res.iloc[index].text_id 
        meta_res_str = df_res.iloc[index][label_col]
        meta_res = json_res_to_struct(meta_res_str)
        meta_acos = get_acos_list(meta_res)

        gpt_4o_res_str = df_vanilla.iloc[text_id][pred_col]
        gpt_4o_res = json_res_to_struct(gpt_4o_res_str)
        gpt_4o_acos = get_acos_list(gpt_4o_res)

        if len(gpt_4o_acos) == 0 or len(meta_acos) == 0:
            print('error index: ', index)
            return 0, 0, 0
        eval_tool = F1ScoreSeq2Seq()
        eval_tool.eval([gpt_4o_acos], [meta_acos])
        return eval_tool.tp, eval_tool.fp, eval_tool.fn
    elif is_sis is True:
        text_id = df_res.iloc[index].text_id 
        meta_res_str = df_res.iloc[index][label_col]
        meta_res = json_res_to_struct(meta_res_str)
        meta_acos = get_acos_list(meta_res)
        meta_acos_sis = get_acos_sis_list(meta_res)

        gpt_4o_res_str = df_vanilla.iloc[text_id][pred_col]
        gpt_4o_res = json_res_to_struct(gpt_4o_res_str)
        gpt_4o_acos = get_acos_list(gpt_4o_res)
        gpt_4o_acos_sis = get_acos_sis_list(gpt_4o_res)

        if len(gpt_4o_acos) == 0 or len(meta_acos) == 0:
            print('error index: ', index)
            return 0, 0, 0
        eval_tool = F1ScoreSeq2Seq()
        mae_list = eval_tool.eval_mae([gpt_4o_acos], [meta_acos], [gpt_4o_acos_sis], [meta_acos_sis])
        return mae_list

def eval_res_vanilla(df_res, df_vanilla, is_sis=False):
    if is_sis is False:
        data = []
        for i in range(df_res.shape[0]):
            tp, fp, fn = eval_instance_vanilla(df_res, df_vanilla, index=i, pred_col='absa_results', label_col='meta_res')
            #tp, fp, fn = eval_instance(df_res, index=i, pred_col=pred_col, label_col=label_col)
            data.append([tp, fp, fn])
            
        df_tmp = pd.DataFrame(data, columns=['tp', 'fp', 'fn'])
        fp = df_tmp['fp'].sum()
        tp = df_tmp['tp'].sum()
        fn = df_tmp['fn'].sum()
        precision = tp / (tp + fp)  
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        acc = tp / (tp + fp + fn)
        return df_tmp, precision, recall, f1, acc, tp, fp, fn
    elif is_sis is True:
        data = []
        for i in range(df_res.shape[0]):
            mae_list = eval_instance_vanilla(df_res, df_vanilla, index=i, pred_col='absa_results', label_col='meta_res', is_sis=True)
            data.append(mae_list)
            
        # flatted the list and get average
        mae_list = [item for sublist in data for item in sublist]
        mae = sum(mae_list) / len(mae_list)
        return data, mae




