""" Thank you Data Scientist @CHRIS DEOTTE, First Method is copied from him.
"""
import itertools
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, gc
import pywt, librosa 


import numpy as np
import librosa
import matplotlib.pyplot as plt

def montage_transform_banana(eeg_50s):
    montage_pairs = {
        'Fp1-F7': ('Fp1', 'F7'),
        'F7-T3': ('F7', 'T3'),
        'T3-T5': ('T3', 'T5'),
        'T5-O1': ('T5', 'O1'),
        'Fp2-F8': ('Fp2', 'F8'),
        'F8-T4': ('F8', 'T4'),
        'T4-T6': ('T4', 'T6'),
        'T6-O2': ('T6', 'O2'),
        'Fp1-F3': ('Fp1', 'F3'),
        'F3-C3': ('F3', 'C3'),
        'C3-P3': ('C3', 'P3'),
        'P3-O1': ('P3', 'O1'),
        'Fp2-F4': ('Fp2', 'F4'),
        'F4-C4': ('F4', 'C4'),
        'C4-P4': ('C4', 'P4'),
        'P4-O2': ('P4', 'O2'),
        'Fz-Cz': ('Fz', 'Cz'),
        'Cz-Pz': ('Cz', 'Pz'),
        'EKG': ('EKG',)  
    }
    
    transformed_eeg = pd.DataFrame()
    
    for new_col, electrodes in montage_pairs.items():
        if len(electrodes) == 1: 
            transformed_eeg[new_col] = eeg_50s[electrodes[0]]
        else:
            transformed_eeg[new_col] = eeg_50s[electrodes[0]] - eeg_50s[electrodes[1]]
    
    return transformed_eeg


def spectrogram_from_eeg_stft_9plu_ekg(data, spe_width = 1024, n_fft=1024, spe_win=512, display=False):
    '''
    data: DataFrame包含19列，前18列为EEG信号，最后一列为EKG信号。
    n_fft, spe_win: STFT的参数。
    display: 是否显示生成的频谱图。
    '''
    n_groups = 9  # 前18个频道分为9组
    img = np.zeros((n_fft//2+1, spe_width, n_groups+1), dtype='float32')  # 最后一张图为EKG
    
    # 处理前18个频道
    for i in range(n_groups):
        group_signals = []
        for j in range(2):  # 每组2个频道
            x = data.iloc[:, i*2+j].to_numpy()
            x = np.nan_to_num(x, nan=np.nanmean(x))
            stft = librosa.stft(y=x, n_fft=n_fft, hop_length=len(x)//spe_width, win_length=spe_win)
            magnitude = np.abs(stft)
            group_signals.append(magnitude)
        
        # 计算每组频道的平均频谱
        group_avg = np.mean(group_signals, axis=0)
        img[:, :, i] = group_avg[:, :spe_width]
    
    # 处理EKG信号
    x_ekg = data.iloc[:, -1].to_numpy()
    x_ekg = np.nan_to_num(x_ekg, nan=np.nanmean(x_ekg))
    stft_ekg = librosa.stft(y=x_ekg, n_fft=n_fft, hop_length=len(x_ekg)//spe_width, win_length=spe_win)
    magnitude_ekg = np.abs(stft_ekg)
    img[:, :, -1] = magnitude_ekg[:, :spe_width]
    
    img = img[:144, :, :]

    if display:
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 调整子图布局
        for i in range(n_groups+1):
            ax = axs.flatten()[i]
            ax.imshow(img[:, :, i], aspect='auto', origin='lower', extent=[0, spe_win, 0, n_fft//2+1])
            title = f'Spectrogram {i+1}' if i < n_groups else 'EKG Spectrogram'
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
        plt.tight_layout()  # 调整整体布局
        plt.show()
        
    return img


def spectrogram_from_eeg_stft_trans_bi(data, spe_width = 1024, n_fft=1024, spe_win = 512, display=False):
    '''用另一种Montage取SAMPLER的准备，transverse_bipolar'''
    img = np.zeros((n_fft//2+1, spe_width, 5), dtype='float32')
    
    signals = []
    
    for k in range(18):
        x = data.iloc[:, k].to_numpy()
        m = np.nanmean(x)
        x = np.nan_to_num(x, nan=m)

        #mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//spe_width, 
                  #n_fft=1024, n_mels=spe_mels, fmin=0, fmax=20, win_length=spe_win)
        mel_spec = librosa.stft(y=x, n_fft=n_fft, hop_length=len(x)//spe_width, win_length=spe_win)
        mel_spec_magnitude = np.abs(mel_spec)
        
        signals.append(mel_spec_magnitude)
    
    # 新的分组和合并逻辑
    # 信号1-3取平均
    group_avg_1_3 = np.mean(np.array(signals[0:3]), axis=0)
    img[:, :, 0] = group_avg_1_3[:, :spe_width]
    
    # 信号4-7、8-11、12-15每组取四个信号的平均
    for i in range(1, 4):
        group_avg = np.mean(np.array(signals[i*4-1:(i+1)*4-1]), axis=0)
        img[:, :, i] = group_avg[:, :spe_width]
    
    # 信号16-18取三个信号的平均
    last_group_avg = np.mean(np.array(signals[15:]), axis=0)
    img[:, :, 4] = last_group_avg[:, :spe_width]

    img = img[:51, :, :]  # 只取一定频率之下的图片
    
    # 可视化部分
    if display:
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # 调整子图布局
        for i in range(5):
            axs[i].imshow(img[:, :, i], aspect='auto', origin='lower', extent=[0, spe_width, 0, 128])
            axs[i].set_title(f'Spectrogram {i+1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Mel Frequency')
        plt.tight_layout()  # 调整整体布局
        plt.show()
        
    return img

def spectrogram_from_eeg_stft(data, spe_width = 1024, n_fft=512, spe_win = 128, display=False):
    '''eeg_data_dict contains contains 19 colomns: 
    'Fp1-F7' 'F7-T3' 'T3-T5' 'T5-O1' (row1)
    'Fp2-F8' 'F8-T4' 'T4-T6' 'T6-O2' (row3)
    'Fp1-F3' 'F3-C3' 'C3-P3' 'P3-O1' (row2)
    'Fp2-F4' 'F4-C4' 'C4-P4' 'P4-O2' (row4)
    'Fz-Cz'
    'Cz-Pz'
    'EKG' '''
    img = np.zeros((n_fft//2+1, spe_width, 5), dtype='float32')
    
    signals = []
    
    for k in range(18):
        x = data.iloc[:, k].to_numpy()
        m = np.nanmean(x)
        x = np.nan_to_num(x, nan=m)

        #mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//spe_width, 
                  #n_fft=1024, n_mels=spe_mels, fmin=0, fmax=20, win_length=spe_win)
        mel_spec = librosa.stft(y=x, n_fft=n_fft, hop_length=len(x)//spe_width, win_length=spe_win)
        mel_spec_magnitude = np.abs(mel_spec)
        
        signals.append(mel_spec_magnitude)
    
    # 根据分组计算平均值并填充到img中
    for i in range(4):  # 处理前16个频道（每组4个频道）
        group_avg = np.mean(np.array(signals[i*4:(i+1)*4]), axis=0)
        group_avg_matched = group_avg[:, :spe_width]
        img[:, :, i] = group_avg_matched

    # 处理最后两个频道（17和18）
    last_group_avg = np.mean(np.array(signals[16:]), axis=0)
    last_group_avg_matched = last_group_avg[:, :spe_width]
    img[:, :, 4] = last_group_avg_matched

    img = img[:51, :, :]
    if display:
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # 调整子图布局
        for i in range(5):
            axs[i].imshow(img[:, :, i], aspect='auto', origin='lower', extent=[0, spe_width, 0, 128,])
            axs[i].set_title(f'Spectrogram {i+1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Mel Frequency')
        plt.tight_layout()  # 调整整体布局
        plt.show()
        
    return img

def spectrogram_from_eeg(data, spe_width = 1000, spe_mels = 128, spe_win = 256, display=False):
    '''eeg_data_dict contains contains 19 colomns: 
    'Fp1-F7' 'F7-T3' 'T3-T5' 'T5-O1' (row1)
    'Fp2-F8' 'F8-T4' 'T4-T6' 'T6-O2' (row3)
    'Fp1-F3' 'F3-C3' 'C3-P3' 'P3-O1' (row2)
    'Fp2-F4' 'F4-C4' 'C4-P4' 'P4-O2' (row4)
    'Fz-Cz'
    'Cz-Pz'
    'EKG' '''
    img = np.zeros((spe_mels, spe_width, 5), dtype='float32')
    
    signals = []
    
    for k in range(18):
        x = data.iloc[:, k].to_numpy()
        m = np.nanmean(x)
        x = np.nan_to_num(x, nan=m)

        mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//spe_width, 
                  n_fft=1024, n_mels=spe_mels, fmin=0, fmax=20, win_length=spe_win)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
        #mel_spec_db = (mel_spec_db + 40) / 40
        signals.append(mel_spec_db)
    
    # 根据分组计算平均值并填充到img中
    for i in range(4):  # 处理前16个频道（每组4个频道）
        group_avg = np.mean(np.array(signals[i*4:(i+1)*4]), axis=0)
        group_avg_matched = group_avg[:, :spe_width]
        img[:, :, i] = group_avg_matched

    # 处理最后两个频道（17和18）
    last_group_avg = np.mean(np.array(signals[16:]), axis=0)
    last_group_avg_matched = last_group_avg[:, :spe_width]
    img[:, :, 4] = last_group_avg_matched

    
    if display:
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # 调整子图布局
        for i in range(5):
            axs[i].imshow(img[:, :, i], aspect='auto', origin='lower', extent=[0, spe_width, 0, spe_mels])
            axs[i].set_title(f'Spectrogram {i+1}')
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Mel Frequency')
        plt.tight_layout()  # 调整整体布局
        plt.show()
        
    return img



def spectrogram_from_eeg_original(parquet_path, display=False):
    NAMES = ['LL','LP','RP','RR']

    FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

    # LOAD MIDDLE 50 SECONDS OF EEG SERIES
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    # VARIABLE TO HOLD SPECTROGRAM
    img = np.zeros((128,256,4),dtype='float32')
    
    if display: plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            # COMPUTE PAIR DIFFERENCES
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            # FILL NANS
            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            # DENOISE
            #if USE_WAVELET:
               # x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            # RAW SPECTROGRAM
            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            # LOG TRANSFORM
            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            # STANDARDIZE TO -1 TO 1
            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        # AVERAGE THE 4 MONTAGE DIFFERENCES
        img[:,:,k] /= 4.0
        
        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            plt.title(f'EEG {parquet_path} - Spectrogram {NAMES[k]}')
            
    if display: 
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f'EEG {parquet_path} Signals')
        plt.show()
        print(); print('#'*25); print()
        
    return img

