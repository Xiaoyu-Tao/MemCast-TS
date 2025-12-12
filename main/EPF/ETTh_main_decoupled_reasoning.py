import pandas as pd
from utils.api_ouput import deepseek_api_output
import json
import os

meaning_dict={'HUFL':'High UseFul Load','HULL':'High UseLess Load','MUFL':'Middle UseFul Load',
              'MULL':'Middle UseLess Load','LUFL':'Low UseFul Load','LULL':'Low UseLess Load','OT':'Oil Temperature'}

def ETTh_main_decoupled_reasoning(data_name,attr,look_back,pred_window,number,api_key,steps_num=3,temperature=0.6,top_p=0.7):

    data_dir = '/dataset/ETT-small/' + data_name + '.csv'
    data = pd.read_csv(data_dir)
    data = data[12 * 30 * 24 + 4 * 30 * 24 - look_back : 12 * 30 * 24 + 8 * 30 * 24]
    date = data.loc[:, 'date'].to_numpy()
    attr_data = data.loc[:, attr].to_numpy()
    data = pd.DataFrame(date, columns=['date'])
    data[attr] = attr_data
    data_lookback = []
    for i in range(10):
        data_lookback.append(data.iloc[i * look_back:(i + 1) * look_back])


    prompt=''
    prompt +='Here is the '+meaning_dict[attr]+' data of the transformer.'
    prompt +=f'I will now give you data for the past {look_back} recorded dates, and please help me forecast the data for next {pred_window} recorded dates.'
    prompt +='But please note that these data will have missing values, so be aware of that.'
    prompt +=f'.Please give me the complete data for the next {pred_window} recorded dates, remember to give me the complete data.'
    prompt +='You must provide the complete data.You mustn\'t omit any content.'
    prompt +='The data is as follows:'
    prompt += data_lookback[number].to_string(index=False)
    prompt +='Please give me the complete data for the next '+str(pred_window)+' recorded dates.'
    prompt +='You should break down the task into four steps, and give me the reasoning process of each step.'
    prompt +='For each short-term prediction, please focus on trends of the input sequence. After one step generation, please take a reflection on your generation.'
    prompt +='And your final answer must follow the format'
    prompt+="""
    <answer>
        \n```\n
        ...
        \n```\n
        </answer>
    Please obey the format strictly. And you must give me the complete answer.
    """

    
    


    with open(f'/output/prompt/{data_name}/prompt_{attr}_{data_name}_{look_back}_{pred_window}_{number}_decoupled_reasoning.txt', 'w') as f:
        f.write(prompt)

    
    model=deepseek_api_output(api_key=api_key,temperature=temperature,top_p=top_p)

    answer=[]

    if os.path.exists(f'/output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}_decoupled_reasoning.json'):
        with open(f'/output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}_decoupled_reasoning.json', 'r') as f:
                answer=json.load(f)
        if len(answer)==3:
            print('This task has been done!')
            return
        else:
            len_answer=len(answer)
    else:
            len_answer=0

    for k in range(3):
        if k<len_answer:
             continue
        else:
            print(f'{k+1} times')
            while True:
                try:
                    reasoning,result=model(prompt)
                    print(reasoning)
                    print(result)
                    break
                except:
                    reasoning='Error'
                    result='Error'
            
            answer.append({'index':k,'reasoning':reasoning,'answer':result})

            with open(f'/output/result/{data_name}/result_{attr}_{data_name}_{look_back}_{pred_window}_{number}_decoupled_reasoning.json', 'w') as f:
                json.dump(answer, f,indent=4)

    print('All done!')