import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime
import pytz
import librosa

st.set_option('deprecation.showPyplotGlobalUse', False)

def find_onsets_in_input_file(in_data, in_freq):
    in_data /= in_data.max() # scaling
    in_data_thresh = np.array(pd.Series(in_data).rolling(int(in_freq / 10)).max().dropna()) # thresholding 
    in_data_thresh[in_data_thresh <= .3] = 0 # remove noise
    idx = np.where(in_data_thresh!=0)[0] # find nnz indices
    segments = np.split(in_data_thresh[idx],np.where(np.diff(idx)!=1)[0]+1) # split around nzz values
    indices = np.split(idx,np.where(np.diff(idx)!=1)[0]+1) # indices of above splits
    indices_max = [index[segment.argmax()] for segment, index in zip(segments, indices)] # get the argmax of every split
    return np.array(indices_max)

def find_onsets_in_metronome_file(metronome_data, metronome_freq):
    return librosa.onset.onset_detect(metronome_data, metronome_freq, units='samples')

def read_audio(path, offset=None, cut_off=None):
    data, freq = librosa.load(path)
    if offset is not None:
        data = data[offset * freq:] # give time to sync to metronome
    elif cut_off is not None:
        data = data[:freq * cut_off]
    return data, freq

def rmse(y, y_hat):
    return np.sqrt(np.sum((y - y_hat) ** 2) / len(y))

def recreate_metronome(in_onsets, metronome_onsets):
    in_onsets_cleaned_short = in_onsets[:len(metronome_onsets)]
    min_rmse = rmse(in_onsets_cleaned_short, metronome_onsets)
    index = 0

    bound = np.abs(in_onsets_cleaned_short[0] - metronome_onsets[0])
    for i in range(-bound, bound, 100):
        _rmse = rmse(in_onsets_cleaned_short, metronome_onsets + i)
        if _rmse < min_rmse:
            min_rmse, index = _rmse, i

    metronome_aligned = metronome_onsets + index 

    mean_step = (metronome_aligned[1:] - metronome_aligned[:-1]).mean()

    metronome_aligned = list(metronome_aligned)

    i = 1
    while len(metronome_aligned) < len(in_onsets):
        metronome_aligned.append(int(metronome_aligned[-1] + mean_step))
        i += 1
    return np.array(metronome_aligned)

def df_columns():
    col_name = ['file']
    for n in range(50):
        column = 'onset_' + str(n+1)
        col_name.append(column)
    col_name.extend(['len_onsets', 'rmse_all_input', 'len_metronome_onsets', 'rmse_w_metronome', 'len__wo_metronome_onsets', 'rmse_wo_metronome'])
    return(col_name)

def metronome_dict(filename, metronome_onsets):
    metronome_dict = {'file':f'metronome:{filename}'}
    m_o_list = list(metronome_onsets)
    for o in m_o_list:
        k = 'onset_' + str(m_o_list.index(o) + 1)
        v = o
        metronome_dict.update({k : v})
    print(metronome_dict) 
    return(metronome_dict)

def audio_dict(filename, in_onsets,len_onsets, rmse_all_input,len_metronome_onsets,rmse_w_metronome,len__wo_metronome_onsets ,rmse_wo_metronome):
    audio_dict = {'file':f'audio:{filename}'}
    a_o_list = list(in_onsets)
    for o in a_o_list:
        k = 'onset_' + str(a_o_list.index(o) + 1)
        audio_dict.update({k : o})
    audio_dict.update({'len_onsets':len_onsets , 'rmse_all_input':rmse_all_input, 'len_metronome_onsets':len_metronome_onsets, 'rmse_w_metronome':rmse_w_metronome, 'len__wo_metronome_onsets':len__wo_metronome_onsets, 'rmse_wo_metronome':rmse_wo_metronome})
    return audio_dict

title = '<p style="font-family:Courier; color:Black; font-size: 60px; font-weight:bold;">TATA recognition</p>'
st.markdown(title, unsafe_allow_html=True)
st.text('Work hard... Play hard')


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.subheader('Step 1: Upload Audio')
upload = st.file_uploader("Upload a audio", type=None, accept_multiple_files=True)
metronome_path = "./data/metronome.wav"

if len(upload) >= 1 :
    df = pd.DataFrame(columns=df_columns())
    timezone = pytz.timezone('Europe/Brussels')
    dateTimeObj = datetime.now(timezone)
    dest_path = dateTimeObj.strftime("%Y%m%d_%H%M%S")    
    for file in upload:
        in_path = file
        results = {}
        # Read file to process and metronome
        in_data, in_freq = read_audio(in_path, offset=2)
        metronome_data, metronome_freq = read_audio(metronome_path, cut_off=5)
        # Remove noise from detected onsets in the input
        in_onsets = find_onsets_in_input_file(in_data, in_freq)
        metronome_onsets = find_onsets_in_metronome_file(metronome_data, metronome_freq)
        # Isolate the part of the input overlapping with the metronome
        # Recreate the metronome signal until the end of the input
        metronome_onsets_full = recreate_metronome(in_onsets, metronome_onsets)
        rmse_total = rmse(in_onsets, metronome_onsets_full)
        rmse_metronome = rmse(in_onsets[:len(metronome_onsets)], metronome_onsets_full[:len(metronome_onsets)])
        rmse_no_metronome = rmse(in_onsets[len(metronome_onsets):], metronome_onsets_full[len(metronome_onsets):])
        #values to be added to the CSV
        len_onsets = len(in_onsets)
        rmse_all_input = rmse_total / in_freq
        len_metronome_onsets = len(metronome_onsets)
        rmse_w_metronome = rmse_metronome / metronome_freq
        len__wo_metronome_onsets = len(in_onsets) - len(metronome_onsets)
        rmse_wo_metronome = rmse_no_metronome / in_freq 
        #extracting filename
        filename = file.name.split('.')[0]
        #adding values
        metro_dict = metronome_dict(filename,metronome_onsets)
        a_dict = audio_dict(filename, in_onsets,len_onsets, rmse_all_input,len_metronome_onsets,rmse_w_metronome,len__wo_metronome_onsets ,rmse_wo_metronome)
        df = df.append(metro_dict, ignore_index = True)
        df = df.append(a_dict, ignore_index = True)
    csv = convert_df(df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=dest_path + '.csv',
        mime='text/csv')
