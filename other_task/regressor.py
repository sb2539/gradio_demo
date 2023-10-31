from gradio.themes.utils import fonts
import torch
import altair as alt
import os
import plotly.graph_objects as go
import graphviz
import seaborn as sns
import torch.nn as nn
from pathlib import Path
graphviz.set_jupyter_format('png')
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.colors as mcolors
import math
import re
import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from scipy import stats
import copy
from sklearn.model_selection import train_test_split
from automl.automl import pycaret_w, h2o_w
from other_task.make_classification_table import classification_table
from gradio.themes.utils import fonts
import pandas as pd
import matplotlib.pyplot as plt



def upload_file_to_dataframe(file):
    column_list =[]
    if file is not None:
        df = pd.read_csv(file.name)  # 업로드된 파일을 DataFrame으로 변환
        for column in df.columns:
            column_list.append(column)
        return gr.DataFrame(value=df, visible=True) , gr.Dropdown(choices=column_list, visible=True), gr.Text(visible=True)
    else:
        return None
def button_visible():
    return gr.Button(visible=True)
# 데이터 분석
def data_ana(data, drop_option):
    #nan_count = data.isna().sum()
    bedrop_col = []
    drop_option = drop_option
    if drop_option == "삭제":
        data = data.dropna()   ## 파이프라인 요소
    for column in data.columns:
        if re.search(r'\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}', str(data[column])):
            data[column] = data[column].apply(lambda x: x.split()[0])
            data['Month'] = pd.to_datetime(data[column]).dt.month
            data['Day'] = pd.to_datetime(data[column]).dt.day
            bedrop_col.append(column)
        elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column])) and len(column) < 5:
            # data[column] = data[column].apply(lambda x: x.split()[0])
            data['Hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
            data['Minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
            data['Second'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.second
            bedrop_col.append(column)
        elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column])) and 'Rise' in column:
            data['Rise_hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
            data['Rise_minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
            bedrop_col.append(column)
        elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column])) and 'Set' in column:
            data['Set_hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
            data['Set_minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
            bedrop_col.append(column)
    return data, gr.Text(visible=True), gr.Text(value=bedrop_col)


with gr.Blocks(title="예측 모델을 만듭니다.", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")),css="temp/style.css") as lt_regression:
    gr.Markdown(
        """
        # 최소한의 선택으로 목적에 맞는 예측 모델을 만들어 드립니다. 
        """
        , elem_id='title'
    )
    task = "Regressor"
    target = []
    target_Train = []
    target_Test = []
    # phase1 upload and data analysis and data preprocessing
    text1 = gr.Text(label = "질문1" ,value="가지고 계신 데이터를 주세요. 하단의 버튼을 누르면 됩니다.")
    upload_button = gr.UploadButton(label="데이터 업로드", file_types=['.pth', '.pt', '.tar', '.csv'], file_count="single",variant="primary")
    raw_data = gr.DataFrame(label="업로드 하신 데이터 입니다", visible=False)
    text2 = gr.Text(label = "질문2",value="예측 대상을 선택해주세요",visible=False)
    target = gr.Dropdown(visible=False, interactive=True) # 타겟 확인
    ## Nan 삭제 부분
    drop_option = gr.Radio(choices=["삭제", "채우기"], visible=False, value="삭제")
    data_ana_done = gr.Text(label = "진행 사항",value="데이터 분석 및 결측치 처리 완료", visible=False)
    ana_drop_btn = gr.Button(value="데이터 분석 및 없는 값 지우기", visible=False)
    be_drop_col = gr.Text(visible=False) # drop 할 column 저장

    upload_button.upload(fn= upload_file_to_dataframe, inputs=[upload_button],outputs=[raw_data, target, text2])
    target.input(fn = button_visible, inputs=None, outputs=[ana_drop_btn])
    ana_drop_btn.click(fn = data_ana, inputs=[raw_data, drop_option], outputs=[raw_data, data_ana_done, be_drop_col])
    lt_regression.queue()