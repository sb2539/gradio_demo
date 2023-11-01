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
from pipeline.class_test import Dataload_anl, Drop_data, Data_preprocess
from automl.automl import AutoMl_library


def upload_file_to_dataframe(file):
    column_list =[]
    if file is not None:
        df = pd.read_csv(file.name)  # 업로드된 파일을 DataFrame으로 변환
        for column in df.columns:
            column_list.append(column)
        return gr.DataFrame(value=df, visible=True) , gr.Dropdown(choices=column_list, visible=True), gr.Text(visible=True)
    else:
        return None

def button_visible(data):
    numeric = data.select_dtypes(include='number').shape[1]
    category = data.select_dtypes(include='object').shape[1]
    return gr.Button(visible=True), gr.Text(visible=True), gr.Radio(visible=True), gr.Text(visible=True, value="수치형 : "+str(numeric)+" 범주형 : "+str(category))

def analysis_variable(select, data):
    if select == "수치형 정보가 기록된 열의 수":
        numeric = data.select_dtypes(include='number').shape[1]
        return numeric
    elif select == "범주형 정보가 기록된 열의 수":
        category = data.select_dtypes(include='object').shape[1]
        return category
# 데이터 분석
def data_ana(data, drop_option, target):
    column_list = []
    data_anal = Dataload_anl(None, drop_option, target)
    target_data = data_anal.get_target(data)
    target_data = pd.DataFrame(target_data)
    dropped_data = data_anal.drop_nan(data)
    new_col_data = data_anal.data_transformation(dropped_data)
    for column in new_col_data.columns:
        column_list.append(column)
    return new_col_data, gr.Text(visible=True), gr.Text(value=column_list), target_data, gr.Button(visible=True)

def data_drop(data, choosed_col):
    drop = Drop_data(choosed_col)
    data = drop.drop_in_column(data)
    return data, gr.Text(visible=True), gr.Slider(visible=True)
def after_slider(split):
    split = split
    return gr.Text(visible=True), gr.Dropdown(visible=True), gr.Button(visible=True), gr.Button(visible=True)

def data_preprocess(data,target_data ,target, split_count, trans_type):
    preprocess = Data_preprocess(trans_type, split_count, target)
    x_train, x_test, y_train, y_test, copy_data = preprocess.data_split(data, target_data)
    return x_train, x_test, y_train, y_test, copy_data, gr.Text(visible=True)
URL = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\"

def run_auto(copy_data, x_train, x_test, y_train, y_test, split, target, auto_lib):
    task = "Regressor"
    automl_lib = AutoMl_library(x_train, x_test, y_train, y_test, split, task, target)
    data = copy_data
    if auto_lib == "모두사용":
        tool_list = ["pycaret", "h2o"]
        print("run")
        for auto in tool_list:
            url = URL + auto+ ".csv"
            if auto == "pycaret":
                pycaret = automl_lib.pycaret_w(data)
                df = pd.read_csv(url)
                yield gr.DataFrame(visible=True, value=df), gr.Dropdown(visible=True)
            elif auto == "h2o":
                h2o = automl_lib.h2o_w(data)
                df = pd.read_csv(url)
                yield gr.DataFrame(visible=True, value=df), gr.Dropdown(visible=True)

def get_board(show_re):
    if show_re == "Pycaret":
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\pycaret.csv"
        df = pd.read_csv(url)
        return df
    elif show_re == "H2o":
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\h2o.csv"
        df = pd.read_csv(url)
        return df


with gr.Blocks(title="예측 모델을 만듭니다.", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")),css="temp/style.css") as lt_regression:
    gr.Markdown(
        """
        # 질문에 답변을 해주시면 필요한 모델을 찾아드립니다. 
        """
        , elem_id='title'
    )
    task = "Regressor"

    # phase1 upload and data analysis and data preprocessing
    text1 = gr.Text(label = "질문1" ,value="가지고 계신 데이터를 주세요. 하단의 버튼을 누르면 업로드 화면이 나타납니다 사용할 데이터를 선택해주세요")
    upload_button = gr.UploadButton(label="데이터 업로드", file_types=['.pth', '.pt', '.tar', '.csv'], file_count="single",variant="primary")
    raw_data = gr.DataFrame(label="업로드 하신 데이터 입니다", visible=False) ## 계속 보여질 데이터
    target_data = gr.DataFrame(visible=False) ## 목표 데이터
    text2 = gr.Text(label = "질문2",value="예측할 목표에 해당하는 열의 이름을 선택해주세요",visible=False)
    target = gr.Dropdown(label="답변 선택",visible=False, interactive=True) # 목표
    col_type = gr.Text(label="데이터에서 수치형 정보가 기록된 열이 몇개고, 범주형 정보가 기록된 열이 몇개인지 보여집니다.",visible=False)
            ## Nan 삭제 부분
    text3 = gr.Text(label= "질문3", value="데이터에 기록이 누락된 값이 있을 수 있습니다. 삭제는 누락된 값이 있는 행을 삭제하고, 채우기는 누락된 값이 있는 열의 평균을 사용합니다. 범주형 열이 있는 경우에는 채우기는 동작하지 않습니다.", visible=False)
    drop_option = gr.Radio(label="답변 선택", choices=["삭제", "채우기"], visible=False)
    ana_drop_btn = gr.Button(value="데이터 분석 및 없는 값 지우기", visible=False)
    data_ana_done = gr.Text(label="진행 사항", value="데이터 분석 및 결측치 처리 완료", visible=False)
    be_drop_col = gr.Text(visible=False) # drop 할 column 저장
    next_process_btn = gr.Button(value="다음 단계로 넘어갑니다.", visible=False)
    # phase2 데이터 전처리
    text4 = gr.Text(label = "질문4",value="현재 데이터에서 불필요한 열이 있나요? 수치값으로 기록된열이 아니라면 제거하는것이 이후의 과정에서 좋습니다.", visible=False)
    check_drop_col = gr.CheckboxGroup(label = "삭제할 열을 선택",visible=False, interactive=True)
    start_drop_btn = gr.Button(value="버튼을 누르면 선택한 열이 데이터에서 지워집니다.", visible=False)
    text5 = gr.Text(label="질문5",value="머신러닝에서는 데이터를 나눠서 학습데이터와 실험데이터를 만듭니다. 아래 막대를 조정해서 실험데이터의 비율을 결정해주세요", visible=False)
    split_slider = gr.Slider(visible=False, value=0.2, label="실험 데이터의 비율을 지정해주세요", minimum=0, maximum=1, step = 0.1)
    text6 = gr.Text(label="질문6",value="데이터 변환을 골라주세요. 데이터 변환은 데이터를 정규분포와 비슷하게 만들어줍니다.", visible=False)
    scaler_type = gr.Dropdown(label="변환선택", choices=["Standard","MinMax", "MaxAbsScaler", "RobustScaler"],visible=False)
    preprocess_btn = gr.Button(value="데이터 전처리 시작", visible=False)
    data_preprocess_done = gr.Text(label="진행상황",value="데이터 전처리 과정이 완료되었습니다.", visible=False)
    next_3phase_btn = gr.Button(value="다음 단계로 넘어갑니다.", visible=False)
    # 데이터 전처리 후 값 저장용 df
    x_train = gr.DataFrame(visible=False)
    x_test = gr.DataFrame(visible=False)
    y_train = gr.DataFrame(visible=False)
    y_test = gr.DataFrame(visible=False)
    copy_data = gr.DataFrame(visible=False)
    ### phase3 AutoML
    text7 = gr.Text(label="질문7", value="머신러닝 모델을 만들기 위해 사용할 도구를 선택해주세요", visible=False)
    auto_lib = gr.Dropdown(label="선택해주세요", choices=["모두사용", "Pycaret", "H2oAutoml"], visible=False, interactive=True)
    auto_btn = gr.Button(value="사용자가 선택한 답변들로 만들수 있는 머신러닝 모델들이 학습됩니다.",visible=False)
    board = gr.DataFrame(visible=False)
    show_re = gr.Dropdown(label = "선택해주세요", choices=["Pycaret", "H2o"], visible=False, interactive=True)
    ### phase1 event
    upload_button.upload(fn= upload_file_to_dataframe, inputs=[upload_button],outputs=[raw_data, target, text2])
    target.input(fn = button_visible, inputs=[raw_data], outputs=[ana_drop_btn, text3, drop_option, col_type])
    ana_drop_btn.click(fn = data_ana, inputs=[raw_data, drop_option, target], outputs=[raw_data, data_ana_done, be_drop_col, target_data, next_process_btn])
    ### phase2 event
    start_drop_btn.click(fn=data_drop, inputs=[raw_data,check_drop_col], outputs=[raw_data, text5, split_slider])
    split_slider.input(fn=after_slider, inputs=[split_slider], outputs=[text6, scaler_type, preprocess_btn, next_3phase_btn])
    preprocess_btn.click(fn = data_preprocess, inputs=[raw_data, target_data, target, split_slider, scaler_type], outputs=[x_train, x_test, y_train, y_test, copy_data, data_preprocess_done])
    ### phase3 event
    auto_btn.click(fn=run_auto, inputs=[copy_data, x_train, x_test, y_train, y_test, split_slider, target, auto_lib], outputs=[board, show_re])
    show_re.input(fn = get_board, inputs=[show_re], outputs=[board])
    lt_regression.queue()


    @next_process_btn.click(inputs=[raw_data],
                     outputs=[text1, upload_button, raw_data, target_data, text2, target, col_type, text3, drop_option, data_ana_done, ana_drop_btn, be_drop_col, next_process_btn,
                             text4, check_drop_col, start_drop_btn, text5, split_slider, text6, scaler_type, data_preprocess_done, next_3phase_btn, x_train, x_test, y_test, y_train, copy_data,
                              preprocess_btn,text7, auto_lib, auto_btn, board, show_re])
    def next_2phase(data):
        column_list = []
        for column in data.columns:
            column_list.append(column)
        return {
            # 1페이즈
            text1 : gr.Text(visible=False),
            upload_button : gr.UploadButton(visible=False),
            raw_data : gr.DataFrame(label="현재 데이터의 모습입니다", visible=True),
            target_data : gr.DataFrame(visible=False),
            text2 : gr.Text(visible=False),
            target : gr.Dropdown(visible=False),  # 타겟 확인
            col_type : gr.Text(visible=False),
            ## Nan 삭제 부분
            text3 : gr.Text(visible=False),
            drop_option : gr.Radio(visible=False),
            data_ana_done : gr.Text(visible=False),
            ana_drop_btn : gr.Button(visible=False),
            be_drop_col : gr.Text(visible=False), # drop 할 column 저장
            next_process_btn : gr.Button(visible=False),
            # 2페이즈
            text4 : gr.Text(visible=True),
            check_drop_col : gr.CheckboxGroup(visible=True, choices=column_list),
            start_drop_btn : gr.Button(visible=True),
            text5 : gr.Text(visible=False),
            split_slider : gr.Slider(visible=False),
            text6 : gr.Text(visible=False),
            scaler_type : gr.Dropdown(visible=False),
            data_preprocess_done : gr.Text(visible=False),
            next_3phase_btn : gr.Button(visible=False),
            x_train : gr.DataFrame(visible=False),
            x_test : gr.DataFrame(visible=False),
            y_train : gr.DataFrame(visible=False),
            y_test : gr.DataFrame(visible=False),
            copy_data : gr.DataFrame(visible=False),
            preprocess_btn : gr.Button(value="데이터 전처리 시작", visible=False),
            text7 : gr.Text(visible=False),
            auto_lib : gr.Dropdown(visible=False),
            auto_btn : gr.Button(visible=False),
            board : gr.DataFrame(visible=False),
            show_re : gr.Dropdown(visible=False)
        }


    @next_3phase_btn.click(inputs=[raw_data],
                            outputs=[text1, upload_button, raw_data, target_data, text2, target, col_type, text3,
                                     drop_option, data_ana_done, ana_drop_btn, be_drop_col, next_process_btn,
                                     text4, check_drop_col, start_drop_btn, text5, split_slider, text6, scaler_type,
                                     data_preprocess_done, next_3phase_btn, x_train, x_test, y_test, y_train, copy_data,preprocess_btn,
                                     text7, auto_lib, auto_btn, board, show_re])
    def next_3phase(data):
        column_list = []
        for column in data.columns:
            column_list.append(column)
        return {
            # 1페이즈
            text1: gr.Text(visible=False),
            upload_button: gr.UploadButton(visible=False),
            raw_data: gr.DataFrame(label="현재 데이터의 모습입니다", visible=False),
            target_data: gr.DataFrame(visible=False),
            text2: gr.Text(visible=False),
            target: gr.Dropdown(visible=False),  # 타겟 확인
            col_type: gr.Text(visible=False),
            ## Nan 삭제 부분
            text3: gr.Text(visible=False),
            drop_option: gr.Radio(visible=False),
            data_ana_done: gr.Text(visible=False),
            ana_drop_btn: gr.Button(visible=False),
            be_drop_col: gr.Text(visible=False),  # drop 할 column 저장
            next_process_btn: gr.Button(visible=False),
            # 2페이즈
            text4: gr.Text(visible=False),
            check_drop_col: gr.CheckboxGroup(visible=False, choices=column_list),
            start_drop_btn: gr.Button(visible=False),
            text5: gr.Text(visible=False),
            split_slider: gr.Slider(visible=False),
            text6: gr.Text(visible=False),
            scaler_type: gr.Dropdown(visible=False),
            data_preprocess_done: gr.Text(visible=False),
            next_3phase_btn: gr.Button(visible=False),
            x_train: gr.DataFrame(visible=False),
            x_test: gr.DataFrame(visible=False),
            y_train: gr.DataFrame(visible=False),
            y_test: gr.DataFrame(visible=False),
            copy_data: gr.DataFrame(visible=False),
            preprocess_btn: gr.Button(value="데이터 전처리 시작", visible=False),
            ### 3페이즈
            text7: gr.Text(visible=True),
            auto_lib: gr.Dropdown(visible=True),
            auto_btn: gr.Button(visible=True),
            board: gr.DataFrame(visible=True),
            show_re: gr.Dropdown(visible=False)
        }
if __name__ == "__main__":
    lt_regression.launch(share=True)