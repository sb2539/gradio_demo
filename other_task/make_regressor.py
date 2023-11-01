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
    if file is not None:
        df = pd.read_csv(file.name)  # 업로드된 파일을 DataFrame으로 변환
        return df
    else:
        return None
def visible(compo):
    compo.visible = True
    return compo
def data_analysis(analysis, data):
    data_des = data.describe(include='all')
    data_des = data_des.reset_index()
    if analysis == "예":
        fig = plt.figure()
        nan_counts = data.isna().sum()
        nan_counts.plot(kind='bar')
        return data, data_des, fig
    elif analysis == "아니오":
        fig = plt.figure()
        data = data.dropna()
        return data, None, fig

def analysis_variable(select, data):
    if select == "수치형 변수 수":
        numeric = data.select_dtypes(include='number').shape[1]
        return numeric
    elif select == "범주형 변수 수":
        category = data.select_dtypes(include='object').shape[1]
        return category
def modi_data(dat_modi, data):
    if dat_modi == "지우기" :
        data = data.dropna()
        return data
    elif dat_modi == "채우기" :
        data = data.fillna(data.mean())
        return data

def make_time_fet(choice,data):
    time_columns = []
    if choice == "예":
        for column in data.columns:
            if re.search(r'\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}', str(data[column])):
                data[column] = data[column].apply(lambda x: x.split()[0])
                data['Month'] = pd.to_datetime(data[column]).dt.month
                data['Day'] = pd.to_datetime(data[column]).dt.day
                time_columns.append(column)
            elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column]))and len(column)<5:
                #data[column] = data[column].apply(lambda x: x.split()[0])
                data['Hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
                data['Minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
                data['Second'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.second
                time_columns.append(column)
            elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column]))and 'Rise' in column:
                data['Rise_hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
                data['Rise_minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
            elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column]))and 'Set' in column:
                data['Set_hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
                data['Set_minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
        return data

def data_drop(check_col, data):
    droped_data = data.drop(columns=check_col)
    return droped_data

def get_column(choice, data):
    column_list = []
    if choice == "예":
        for column in data.columns:
            column_list.append(column)
        return column_list
def transformation(column, data):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 8))

    pd.DataFrame(data[column]).hist(ax=ax1, bins=50)
    pd.DataFrame((data[column] + 1).transform(np.log)).hist(ax=ax2, bins=50)
    pd.DataFrame(stats.boxcox(data[column] + 1)[0]).hist(ax=ax3, bins=50)
    pd.DataFrame(StandardScaler().fit_transform(np.array(data[column]).reshape(-1, 1))).hist(ax=ax4, bins=50)
    pd.DataFrame(MinMaxScaler().fit_transform(np.array(data[column]).reshape(-1, 1))).hist(ax=ax5, bins=50)

    ax1.set_ylabel('Normal')
    ax2.set_ylabel('Log')
    ax3.set_ylabel('Box Cox')
    ax4.set_ylabel('Standard')
    ax5.set_ylabel('MinMax')
    return fig

def set_trans(column, way, data):
    if way == 'Log':
        data[column] = (data[column]+1).transform(np.log)
        return data
    elif way == 'BoxCox':
        data[column] = stats.boxcox(data[column]+1)[0]
        return data
    elif way == 'Standard':
        data[column] = StandardScaler().fit_transform(np.array(data[column]).reshape(-1,1))
        return data
    elif way == 'MinMax':
        data[column] = MinMaxScaler().fit_transform(np.array(data[column]).reshape(-1,1))
        return data
def get_target(choice_col, data):
    copy_data = copy.deepcopy(data)
    input_feature = copy.deepcopy(data)
    target = input_feature[choice_col]
    target = pd.DataFrame(target)
    input_feature = input_feature.drop(choice_col, axis = 1)
    return data,target ,copy_data, input_feature

def split_data(split, data, target):
    xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=split, random_state=1)
    scaler = StandardScaler()
    ytrain = pd.DataFrame(ytrain)
    ytest = pd.DataFrame(ytest)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.fit_transform(xtest)
    print("split_done")
    return xtrain, xtest, ytrain, ytest
URL = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\"
def call_auto_re(target, copy_data, split_bar, tool):
    tool_list = ["Pycaret", "H2o"]
    if tool == "모두":
        for it_tool in tool_list:
            #url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\pycaret.csv"
            url = URL + it_tool+ ".csv"
            task = "Regressor"
            print("run")
            if it_tool == "Pycaret":
                pycaret_w(task, target, split_bar, copy_data)
                df = pd.read_csv(url)
                yield df
            elif it_tool =="H2o":
                h2o_w(task, target, split_bar, copy_data)
                df = pd.read_csv(url)
                yield df
    elif tool == "Pycaret":
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\pycaret.csv"
        task = "Regressor"
        pycaret_w(task, target, split_bar, copy_data)
        df = pd.read_csv(url)
        return df
    elif tool == "H2o":
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\h2o.csv"
        task = "Regressor"
        h2o_w(task, target, split_bar, copy_data)
        df = pd.read_csv(url)
        return df
    else :
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\pycaret.csv"
        task = "Regressor"
        print("none")
        pycaret_w(task, target, split_bar, copy_data)
        df = pd.read_csv(url)
        return df

def read_csv(tool):
    if tool == "Pycaret":
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\pycaret.csv"
        df = pd.read_csv(url)
        return df
    elif tool == "H2o":
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\h2o.csv"
        df = pd.read_csv(url)
        return df


# 회귀 인터페이스
with gr.Blocks(title="예측 모델을 만듭니다.", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")),css="temp/style.css") as regression:
    gr.Markdown(
        """
        # 지금부터 예측 모델 만드는 과정을 도와드릴게요
        """
        , elem_id='title'
    )
    task = "Regressor"
    target = []
    target_Train = []
    target_Test = []
    with gr.Row():
        with gr.Column(scale=1):
            # phase 1: 데이터 입력, 데이터 분석
            text1=gr.Text(label="질문 1", value="가지고 계신 데이터가 있으면 아래 버튼을 눌러주세요")
            upload_button = gr.UploadButton(label="데이터 업로드", file_types=['.pth', '.pt', '.tar', '.csv'],file_count="single",variant="primary")
            text2=gr.Text(label="질문 2", value="데이터의 정보를 확인하실건가요? 이 과정은 기록 되지 않은 값이 있나 확인하는 과정으로 아니오를 선택하시면 아래 버튼 클릭 후 질문 4로 넘어가주세요")
            dat_anlysis = gr.Radio(label="선택", choices=["예", "아니오"])
            anl_btn = gr.Button(value="버튼을 누르면 실행됩니다.")
            text3=gr.Text(label="질문 3", value="2번째 질문에서 예를 선택한 경우만 해당됩니다, 비어있는 값이 오른쪽 그래프에 있나요? 빈 값을 지우거나 채울 수 있습니다. 선택해주세요")
            dat_modi = gr.Radio(label="선택", choices=["지우기", "채우기"])
            modi_btn = gr.Button(value="버튼을 누르면 원하는 작업이 실행됩니다")
            text4=gr.Text(label="질문 4", value="오른쪽표를 봤을때 12:13:12 나 9/23/2015 같은 형태로 기록된 데이터가 있다면 예를 눌러주세요")
            time_decom = gr.Radio(label="선택", choices=["예", "아니오"])
            decom_btn = gr.Button(value="작업이 실행됩니다")
            # phase 2 : 데이터 전처리 (삭제)
            text5 = gr.Text(label="질문 5", value="데이터에서 필요없는 정보가 있나요? 있다면 선택해주세요 만약 시간과 관련된 정보가 있다면 제거해주세요 예시 : ~Time", visible=False)
            drop_check = gr.CheckboxGroup(label="필요없는 열 선택", choices=["none"], visible=False)
            drop_btn = gr.Button(value="선택한 열이 제거됩니다.",visible=False)
            # phase 3 : 데이터 전처리 (변환, 타겟 설정, split)
            text6 = gr.Text(label="질문 6", value="데이터의 각각의 열에 변환을 적용할까요? 아니오를 선택하시면 질문 8로 바로 넘어가세요.", visible=False)
            tran_sel = gr.Radio(label="선택", choices=["예", "아니오"], visible=False)
            text7 = gr.Text(label="질문 7", value="변환을 적용할 열과, 적용할 변환을 선택해주세요", visible=False)
            tran_col = gr.Dropdown(visible=False, interactive=True) # 변환 적용할 열
            tran_way = gr.Dropdown(visible=False, interactive=True, choices=['Normal','Log','BoxCox','Standard','MinMax']) # 적용할 변환 방법
            tran_btn = gr.Button(value="선택을 적용합니다.", visible=False)
            ##### 내일 해야 하는 부분
            text8 = gr.Text(label="질문 8", value="데이터에서 어떤 정보를 예측하고자 하나요?",visible=False)
            choose_target = gr.Dropdown(visible=False, label="예측할 열을 선택해주세요") ## target
            target_btn = gr.Button(visible=False, value="예측할 열을 적용합니다.")
            text9 = gr.Text(label="질문 9", value="머신러닝에서는 데이터를 학습데이터와 실험데이터로 나눠줘야 합니다. 아래에서 비율을 선택해주세요", visible=False)
            split_bar = gr.Slider(visible=False, value=0.2, label="실험 데이터의 비율을 지정해주세요", minimum=0, maximum=1)
            split_btn = gr.Button(visible=False, value="데이터를 나눕니다.")
        with gr.Column(scale=1):
            # 데이터 입력 파트
            raw_data = gr.DataFrame(visible=True, label="데이터 입니다.")
            select_plot = gr.Radio(label="확인하고 싶은 항목을 골라주세요 ", choices=["수치형 변수 수", "범주형 변수 수", "확인 안함"],value="그래프 없음")
            variable_num = gr.Text(label="개수")
            des_data = gr.DataFrame(visible=True, label="데이터의 통계정보가 보여집니다.")
            ana_plot = gr.Plot(label="비어있는 값의 그래프가 보여집니다.")
            # select_plot = gr.Radio(label="확인하고 싶은 항목을 골라주세요 ", choices=["수치형 변수 수", "범주형 변수 수", "확인 안함"], value="그래프 없음")
            # variable_num = gr.Text(label="개수")
            # phase 2 : 데이터 삭제부터
            # phase 3 : 데이터 변환부터 split 까지
            tran_drop_down = gr.Dropdown(visible=False, interactive=True)
            tran_plot = gr.Plot(label="위 드롭박스에서 확인하고 싶은 열을 선택하세요, 시간과 관련 없는 열을 선택해주세요",visible=False)
            #######
            copy_data = gr.DataFrame(visible=False, label="복사된 데이터 입니다.")
            input_feature = gr.DataFrame(visible=False, label="모델 입력 특징")
            target_data = gr.DataFrame(visible=False, label="타겟으로 분리된 데이터 입니다.")
            target_train = gr.DataFrame(visible=False, label="타겟 학습 데이터 입니다.")
            target_test = gr.DataFrame(visible=False, label="타겟 테스트 데이터입니다.")
            train_data = gr.DataFrame(visible=False, label="학습 데이터 입니다.")
            test_data = gr.DataFrame(visible=False, label="실험 데이터 입니다.")

    # 데이터 입력 파트
    upload_button.upload(fn= upload_file_to_dataframe, inputs=[upload_button],outputs=[raw_data])
    anl_btn.click(fn=data_analysis, inputs=[dat_anlysis, raw_data], outputs=[raw_data, des_data, ana_plot])
    select_plot.change(fn =analysis_variable, inputs=[select_plot, raw_data], outputs=[variable_num])
    modi_btn.click(fn=modi_data, inputs = [dat_modi, raw_data], outputs=[raw_data])
    decom_btn.click(fn= make_time_fet,inputs = [time_decom, raw_data], outputs=[raw_data])
    next1_btn = gr.Button(value="다음 과정으로 넘어갑니다")
    # phase 2 : 데이터 삭제
    drop_btn.click(fn=data_drop, inputs=[drop_check, raw_data], outputs=[raw_data])
    tran_btn.click(fn=get_column, inputs=[tran_sel, raw_data], outputs=[tran_drop_down])
    next2_btn = gr.Button(value="다음 과정", visible=False)
    # phase 3 : 데이터 변환부터 split 까지
    tran_drop_down.input(fn = transformation, inputs=[tran_drop_down, raw_data], outputs=tran_plot)
    tran_btn.click(fn=set_trans, inputs=[tran_col,tran_way,raw_data],outputs=[raw_data])
    ### 새로 추가
    target_btn.click(fn=get_target, inputs=[choose_target, raw_data], outputs=[raw_data,target_data,copy_data, input_feature])
    split_btn.click(fn = split_data, inputs=[split_bar,raw_data, target_data], outputs=[train_data, test_data, target_train, target_test])
    next3_btn = gr.Button(value="지금 부터 머신러닝 모델을 만듭니다.", visible=False)
    ### phase 4 : 머신러닝 모델
    text10 = gr.Text(label="질문 10", value="아래에서 머신러닝 모델을 만들 도구를 선택해주세요. 이 작업은 오래걸립니다.", visible=False)
    choose_tool = gr.Radio(label="AutoML도구 선택", choices=["모두", "Pycaret", "H2o"], visible=False)
    train_btn = gr.Button("머신 러닝 모델 만들기 시작", visible=False)
    tool_drop = gr.Dropdown(label="확인하고 싶은 도구를 선택해주세요, 그래프가 생기지 않는다면 아직 학습중입니다.",visible=False, choices=["Pycaret", "H2o"] ,interactive=True)
    dash_board = gr.DataFrame(label="머신러닝 대쉬보드", visible=False, every=5)
    train_btn.click(fn=call_auto_re, inputs=[choose_target, copy_data, split_bar, choose_tool], outputs=[dash_board])
    tool_drop.input(fn=read_csv, inputs=[tool_drop], outputs=[dash_board])
    regression.queue()
    #tran_btn.click(fn=transformation, inputs=[tran_drop_down, raw_data], outputs=[tran_plot])
    #drop_btn.click

    @next1_btn.click(inputs=[raw_data],
                          outputs=[text1, text2, text3, text4, upload_button, dat_anlysis, anl_btn, dat_modi, modi_btn,raw_data ,time_decom, decom_btn, des_data, ana_plot, select_plot, variable_num,next1_btn,
                                   next2_btn, text5, drop_check, drop_btn, text6, tran_sel, tran_btn, tran_drop_down, tran_plot,
                                   tran_col, tran_way, text7, text8,choose_target, text9, split_bar, split_btn,target_btn, copy_data,input_feature ,train_data, test_data, target_data, target_train, target_test,
                                   next3_btn, text10,choose_tool ,train_btn,tool_drop,dash_board])
    def next2_phase(data):
        column_list = []
        for column in data.columns:
            column_list.append(column)
        return {
            # 1페이즈
            text1 :gr.Text(visible=False),
            upload_button : gr.UploadButton(visible=False),
            text2 : gr.Text(visible=False),
            dat_anlysis : gr.Radio(visible=False),
            anl_btn : gr.Button(visible=False),
            text3 : gr.Text(visible=False),
            dat_modi : gr.Radio(visible=False),
            modi_btn : gr.Button(visible=False),
            text4 : gr.Text(visible=False),
            time_decom : gr.Radio(visible=False),
            decom_btn : gr.Button(visible=False),
            raw_data : gr.DataFrame(visible=True, label="데이터 입니다."),
            des_data : gr.DataFrame(visible=False),
            ana_plot : gr.Plot(visible=False),
            select_plot : gr.Radio(visible=False),
            variable_num : gr.Text(visible=False),
            next1_btn : gr.Button(visible=False),
            # 2페이즈 기능
            next2_btn : gr.Button(visible=True),
            text5 : gr.Text(label="질문 5", value="데이터에서 필요없는 정보가 있나요? 있다면 선택해주세요", visible=True),
            drop_check : gr.CheckboxGroup(label="필요없는 열 선택", choices=column_list, visible=True),
            drop_btn : gr.Button(value="선택한 열이 제거됩니다.", visible=True),
            # 3페이즈 기능
            text6 : gr.Text(label="질문 6", visible=False),
            tran_sel : gr.Radio(label="선택", choices=["예", "아니오"], visible=False),
            tran_btn : gr.Button(value="선택을 적용합니다.", visible=False),
            tran_drop_down : gr.Dropdown(visible=False, choices=column_list),
            tran_plot : gr.Plot(label="위 드롭박스에서 확인하고 싶은 열을 선택하세요, 시간과 상관없는 열을 선택해주세요", visible=False),
            text7 : gr.Text(label="질문 7", value="변환을 적용할 열과, 적용할 변환을 선택해주세요", visible=False),
            tran_col : gr.Dropdown(visible=False, interactive=True),  # 변환 적용할 열
            tran_way : gr.Dropdown(visible=False, interactive=True),  # 적용할 변환 방법
            #####
            text8 : gr.Text(visible=False),
            choose_target : gr.Dropdown(visible=False),
            target_btn : gr.Button(visible=False),
            text9 : gr.Text(visible=False),
            split_bar : gr.Slider(visible=False),
            split_btn : gr.Button(visible=False),
            copy_data : gr.DataFrame(visible=False),
            input_feature : gr.DataFrame(visible=False, label="모델 입력 특징"),
            train_data : gr.DataFrame(visible=False),
            test_data : gr.DataFrame(visible=False),
            target_data : gr.DataFrame(visible=False),
            target_train : gr.DataFrame(visible=False),
            target_test : gr.DataFrame(visible=False),
            next3_btn : gr.Button(visible=False),
            ###phase 4
            text10 : gr.Text(visible=False),
            choose_tool : gr.Radio(visible=False),
            train_btn : gr.Button(visible=False),
            tool_drop : gr.Dropdown(visible=False),
            dash_board : gr.DataFrame(visible=False)
    }


    @next2_btn.click(inputs=[raw_data],
                     outputs=[text1, text2, text3, text4, upload_button, dat_anlysis, anl_btn, dat_modi, modi_btn,
                              raw_data, time_decom, decom_btn, des_data, ana_plot, select_plot, variable_num, next1_btn,
                              next2_btn, text5, drop_check, drop_btn, text6, tran_sel, tran_btn, tran_drop_down,
                              tran_plot, tran_col, tran_way,text7,text8, choose_target, text9, split_bar, split_btn,target_btn,train_data, test_data, copy_data,input_feature, target_data, target_train, target_test,
                              next3_btn, text10,choose_tool ,train_btn,tool_drop ,dash_board])
    def next3_phase(data):
        column_list = []
        for column in data.columns:
            column_list.append(column)
        return {
            # 1페이즈
            text1: gr.Text(visible=False),
            upload_button: gr.UploadButton(visible=False),
            text2: gr.Text(visible=False),
            dat_anlysis: gr.Radio(visible=False),
            anl_btn: gr.Button(visible=False),
            text3: gr.Text(visible=False),
            dat_modi: gr.Radio(visible=False),
            modi_btn: gr.Button(visible=False),
            text4: gr.Text(visible=False),
            time_decom: gr.Radio(visible=False),
            decom_btn: gr.Button(visible=False),
            raw_data: gr.DataFrame(visible=True, label="데이터 입니다."),
            des_data: gr.DataFrame(visible=False),
            ana_plot: gr.Plot(visible=False),
            select_plot: gr.Radio(visible=False),
            variable_num: gr.Text(visible=False),
            next1_btn: gr.Button(visible=False),
            # 2페이즈 기능
            next2_btn: gr.Button(visible=False),
            text5: gr.Text(label="질문 5", value="데이터에서 필요없는 정보가 있나요? 있다면 선택해주세요", visible=False),
            drop_check: gr.CheckboxGroup(label="필요없는 열 선택", choices=column_list, visible=False),
            drop_btn: gr.Button(value="선택한 열이 제거됩니다.", visible=False),
            # 3페이즈 기능
            text6: gr.Text(label="질문 6", visible=True),
            tran_sel: gr.Radio(label="선택", choices=["예", "아니오"], visible=True),
            tran_btn: gr.Button(value="선택을 적용합니다.", visible=True),
            tran_drop_down: gr.Dropdown(visible=True, choices=column_list),
            tran_plot: gr.Plot(label="위 드롭박스에서 확인하고 싶은 열을 선택하세요, 시간과 상관없는 열을 선택해주세요", visible=True),
            text7: gr.Text(label="질문 7", value="변환을 적용할 열과, 적용할 변환을 선택해주세요", visible=True),
            tran_col : gr.Dropdown(visible=True, interactive=True, choices=column_list),  # 변환 적용할 열
            tran_way : gr.Dropdown(visible=True, interactive=True),  # 적용할 변환 방법
            text8: gr.Text(visible=True),
            choose_target: gr.Dropdown(visible=True, choices=column_list),
            target_btn: gr.Button(visible=True),
            text9: gr.Text(visible=True),
            split_bar: gr.Slider(visible=True),
            split_btn: gr.Button(visible=True),
            copy_data: gr.DataFrame(visible=False),
            input_feature: gr.DataFrame(visible=False, label="모델 입력 특징"),
            train_data: gr.DataFrame(visible=False),
            test_data: gr.DataFrame(visible=False),
            target_data: gr.DataFrame(visible=False),
            target_train: gr.DataFrame(visible=False),
            target_test: gr.DataFrame(visible=False),
            next3_btn: gr.Button(visible=True),
            ## phase 4
            text10: gr.Text(visible=False),
            choose_tool: gr.Radio(visible=False),
            train_btn: gr.Button(visible=False),
            tool_drop: gr.Dropdown(visible=False),
            dash_board: gr.DataFrame(visible=False)
        }
    @next3_btn.click(inputs=[raw_data],
                     outputs=[text1, text2, text3, text4, upload_button, dat_anlysis, anl_btn, dat_modi, modi_btn,
                              raw_data, time_decom, decom_btn, des_data, ana_plot, select_plot, variable_num, next1_btn,
                              next2_btn, text5, drop_check, drop_btn, text6, tran_sel, tran_btn, tran_drop_down,
                              tran_plot, tran_col, tran_way,text7,text8, choose_target, text9, split_bar, split_btn,target_btn ,train_data, test_data, copy_data,input_feature, target_data, target_train, target_test,
                              next3_btn, text10,choose_tool ,train_btn,tool_drop ,dash_board])
    def next4_phase(data):
        column_list = []
        for column in data.columns:
            column_list.append(column)
        return {
            # 1페이즈
            text1: gr.Text(visible=False),
            upload_button: gr.UploadButton(visible=False),
            text2: gr.Text(visible=False),
            dat_anlysis: gr.Radio(visible=False),
            anl_btn: gr.Button(visible=False),
            text3: gr.Text(visible=False),
            dat_modi: gr.Radio(visible=False),
            modi_btn: gr.Button(visible=False),
            text4: gr.Text(visible=False),
            time_decom: gr.Radio(visible=False),
            decom_btn: gr.Button(visible=False),
            raw_data: gr.DataFrame(visible=False, label="데이터 입니다."),
            des_data: gr.DataFrame(visible=False),
            ana_plot: gr.Plot(visible=False),
            select_plot: gr.Radio(visible=False),
            variable_num: gr.Text(visible=False),
            next1_btn: gr.Button(visible=False),
            # 2페이즈 기능
            next2_btn: gr.Button(visible=False),
            text5: gr.Text(label="질문 5", value="데이터에서 필요없는 정보가 있나요? 있다면 선택해주세요", visible=False),
            drop_check: gr.CheckboxGroup(label="필요없는 열 선택", choices=column_list, visible=False),
            drop_btn: gr.Button(value="선택한 열이 제거됩니다.", visible=False),
            # 3페이즈 기능
            text6: gr.Text(label="질문 6", visible=False),
            tran_sel: gr.Radio(label="선택", choices=["예", "아니오"], visible=False),
            tran_btn: gr.Button(value="선택을 적용합니다.", visible=False),
            tran_drop_down: gr.Dropdown(visible=False, choices=column_list),
            tran_plot: gr.Plot(label="위 드롭박스에서 확인하고 싶은 열을 선택하세요, 시간과 상관없는 열을 선택해주세요", visible=False),
            text7: gr.Text(label="질문 7", value="변환을 적용할 열과, 적용할 변환을 선택해주세요", visible=False),
            tran_col : gr.Dropdown(visible=False, interactive=True, choices=column_list),  # 변환 적용할 열
            tran_way : gr.Dropdown(visible=False, interactive=True),  # 적용할 변환 방법
            text8: gr.Text(visible=False),
            choose_target: gr.Dropdown(visible=False, choices=column_list),
            target_btn: gr.Button(visible=False),
            text9: gr.Text(visible=False),
            split_bar: gr.Slider(visible=False),
            split_btn: gr.Button(visible=False),
            copy_data: gr.DataFrame(visible=False),
            input_feature: gr.DataFrame(visible=False, label="모델 입력 특징"),
            train_data: gr.DataFrame(visible=False),
            test_data: gr.DataFrame(visible=False),
            target_data: gr.DataFrame(visible=False),
            target_train: gr.DataFrame(visible=False),
            target_test: gr.DataFrame(visible=False),
            next3_btn: gr.Button(visible=False),
            ## phase 4
            text10: gr.Text(visible=True),
            choose_tool: gr.Radio(visible=True),
            train_btn: gr.Button(visible=True),
            tool_drop: gr.Dropdown(visible=True),
            dash_board: gr.DataFrame(visible=True)
        }