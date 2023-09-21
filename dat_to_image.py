import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import pywt
import os
import warnings
#본 파일은 .dat파일을 읽고 이를 par#, video#, session#, channel#, features, vlence, arousal이 column인 DF를 만들도록 한다.
#fix participant

#Referencing Setting
reference_mode = 'Laplace'
# reference_mode = 'None'
# reference_mode = 'CAR'

#상수
VIDEO = 40
PARTICIPANT = 32
CHANNEL = 32
SESSION = 63
TWENTE_END = 22
# GENEVA_START = 23
LENGTH = 9
AVOID_INF = 1e-6 
#Channel Image mapping table, height * width
twente_table = [[0,	0,	1,	1,	0,	30,	30,	0,	0],
                [0,	0,	2,	2,	0,	29,	29,	0,	0],
                [3,	0,	4,	0,	31,	0,	27,	0,	28],
                [0,	6,	0,	5,	0,	26,	0,	25,	0],
                [7,	0,	8,	0,	32,	0,	23,	0,	24],
                [0,	10,	0,	9,	0,	22,	0,	21,	0],
                [11, 0,	12,	0,	13,	0,	19,	0,	20],
                [0,	0,	0,	14,	0,	18,	0,	0,	0],
                [0,	0,	15,	15,	12,	17,	17,	0,	0]]

geneva_table = [[0,	0,	1,	1,	0,	20,	20,	0,	0],
                [0,	0,	2,	2,	0,	19,	19,	0,	0],
                [4,	0,	3,	0,	18,	0,	21,	0,	22],
                [0,	5,	0,	6,	0,	24,	0,	23,	0],
                [8,	0,	7,	0,	17,	0,	26,	0,	32],
                [0,	9,	0,	10,	0,	25,	0,	28,	0],
                [12, 0,	11,	0,	14,	0,	31,	0, 27],
                [0,	0,	0,	15,	0,	29,	0,	0,	0],
                [0,	0,	16,	16,	11,	30,	30,	0,	0]]
#one row dataframe을 32개의 row로 복제시킵니다.
def fit_rows_to_session(dataframe):
    one_row_dataframe = dataframe.copy()

    for _ in range(SESSION-1):
        dataframe = pd.concat([dataframe,one_row_dataframe], ignore_index=True)
    
    return dataframe

def WE(y, level = 5, wavelet = 'db4'):
    from math import log

    n = len(y)
    #ap : amplitude
    ap = {}

    for lev in range(0,level):
        (y, cD) = pywt.dwt(y, wavelet)
        ap[lev] = y

    # Energy

    Enr = np.zeros(level)
    # print(ap)
    for lev in range(0,level):
        Enr[lev] = np.sum(np.power(ap[lev],2))/n

    #Energy_total
    Et = np.sum(Enr)

    #각 wavlet의 에너지 비율
    Pi = np.zeros(level)
    for lev in range(0,level):
            if reference_mode == 'Laplace':
                Pi[lev] = Enr[lev]/(Et + AVOID_INF)
            else:
                Pi[lev] = Enr[lev]/Et
            
    # print(Pi, np.log(Pi))

    #wavelet entropy
    we = - np.sum(np.dot(Pi ,np.log(Pi+AVOID_INF)))

    return np.mean(Enr), np.mean(Pi), we

##Some Referencing Techniques
def create_dir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

def CAR_Referencing(EEG_data):
    # print(EEG_data.shape)
    temp_EEG_data = EEG_data
    #shpae(1,128)
    average_EEG_data = EEG_data.mean(axis=0)

    #change shape to (32. 128), copy and paste
    average_EEG_data = average_EEG_data.reshape(1,128)
    average_EEG_data = np.repeat(average_EEG_data, CHANNEL, axis = 0)

    temp_EEG_data = EEG_data - average_EEG_data
    
    return temp_EEG_data

def Laplacian_Referencing(EEG_data, particiapnt_id):
    
    temp_EEG_data = np.zeros((32,128))
    if particiapnt_id <=TWENTE_END:
        referencing_table = np.array(twente_table)
    else:
        referencing_table = np.array(geneva_table)

    for height in range(LENGTH):
        for weigh in range(LENGTH):
            #index는 0부터 시작하므로 1뺌
            center_channel_index = referencing_table[height,weigh] - 1
            #index가 -1은 빈 데이터
            if center_channel_index <0:
                continue

            #index extraction
            surrounding_channel_index_list =[]
            if weigh != 0:
                surrounding_channel_index_list.append(referencing_table[height, (weigh-1)]-1)
            if weigh != LENGTH - 1:
                surrounding_channel_index_list.append(referencing_table[height, (weigh+1)]-1)
            if height != 0:
                surrounding_channel_index_list.append(referencing_table[(height-1), weigh]-1)
            if height != LENGTH -1:
                surrounding_channel_index_list.append(referencing_table[(height+1), weigh]-1)
            
            if -1 in surrounding_channel_index_list:
                surrounding_channel_index_list.remove(-1)

            # print(surrounding_channel_index_list)
            refereced_EEG = EEG_data[center_channel_index] - np.mean(EEG_data[surrounding_channel_index_list], axis = 0)

            np.put(temp_EEG_data, center_channel_index, refereced_EEG)

    return temp_EEG_data
            




##처리할 이미지를 저장하기 위한 폴더 생성
create_dir("./Data/"+reference_mode+"_data_palettes_test/")
create_dir("./Data/"+reference_mode+"_data_palettes_train/")

#Start making
for participant_id in tqdm(range(1,33)):

    filepath = "./Data/raw_data/s" + format(participant_id, '02') +".dat"

    with open(filepath, 'rb') as f:
        x_dict = pickle.load(f, encoding='latin1')

    #analyze unknown dictionary
    # print(x_dict.keys())

    #(40,4) (video_#, label_#) : (valence, arousal, dominance, liking)
    video_labels = x_dict["labels"]

    #(40,40,8064) (video_#, channel#, EEG)
    video_data = x_dict["data"]

    #각 particpant의 데이터가 저장될 빈 데이터 프레임
    participant_dataframe = pd.DataFrame()

    #fix video_num
    for video_num in range(1,VIDEO+1):
        
        ####PART1.#####  Valence Arousal Data
        Val_Aro_value_ndarray = video_labels[video_num-1,[0,1]]
        Val_Aro_value_list = Val_Aro_value_ndarray.tolist()

        valence = Val_Aro_value_list[0]
        arousal = Val_Aro_value_list[1]

        if arousal >=5:
            if valence >= 5:
                palette_name = 'HAHV'
            else:
                palette_name = 'HALV'
        else:
            if valence >= 5:
                palette_name = 'LAHV'
            else:
                palette_name = 'LALV'

        ####PART2.#####  Palette Making

        session_palette_df = pd.DataFrame()
        #fix session
        for session_num in range(1,64):
            #numpy array, (32,128) : (channel, EEG)
            session_data = video_data[video_num-1 , 0:CHANNEL, 128*(session_num-1) : 128*session_num]

            #Referencing Filter('CAR', 'Laplace')
            if reference_mode == 'Laplace':
                session_data = Laplacian_Referencing(EEG_data=session_data, particiapnt_id = participant_id)
            if reference_mode =='CAR':
                session_data = CAR_Referencing(EEG_data = session_data)

            # [0] : Empty, [range(1,33)] : corresponding features to channel
            features_in_one_session = [[0,0,0]]

            #store features of one seesion
            for channel_num in range(1,CHANNEL+1):
                we,m_wer,m_enr = WE(session_data[channel_num-1])
                features_in_one_session.append([we,m_wer,m_enr])
            
            session_palette =[]
            for height in range(LENGTH):

                width_temp = []
                for width in range(LENGTH):
                    #TWENTE, GENVA 전극의 위치와 번호의 차이 고려
                    if participant_id <= TWENTE_END:
                        width_temp.append(features_in_one_session[twente_table[height][width]])
                    #GENEVA
                    else:
                        width_temp.append(features_in_one_session[geneva_table[height][width]])
                
                session_palette.append(width_temp)

            session_palette_ndarray = np.array(session_palette)

            # Resize the image to the desired shape (224, 224, 3)
            # resized_palette_ndarray = resize(session_palette_ndarray, (224, 224, 3), anti_aliasing=True)

            if participant_id <=24:
                pickle.dump(session_palette_ndarray, open("./Data/"+reference_mode+"_data_palettes_train/" + 'Participant_' + str(participant_id) + '_Video_'+str(video_num)+'_Session_'+str(session_num)+'_'+palette_name + '.pkl', 'wb'))
            else:
                pickle.dump(session_palette_ndarray, open("./Data/"+reference_mode+"_data_palettes_test/" + 'Participant_' + str(participant_id) + '_Video_'+str(video_num)+'_Session_'+str(session_num)+'_'+palette_name + '.pkl', 'wb'))






