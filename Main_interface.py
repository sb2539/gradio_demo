import matplotlib.pyplot as plt
import pandas as pd
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
#TODO: CSS파일 넣어서 헤더같은 것들 꾸미기
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
# from other_task.make_classification_table import classification_table
# from other_task.make_regressor import regression
# from other_task.make_classification import classification
# from other_task.make_lite_regressor import lt_regression

def hint1_image(hint1_option):
    path = ""
    if hint1_option == "이미지데이터":
        path = "image/main/image.JPG"
    elif hint1_option == "테이블데이터":
        path = "image/main/tabular.JPG"
    elif hint1_option == "시계열데이터":
        path = "image/main/timeseries.JPG"
    elif hint1_option == "혼합데이터":
        path = "image/main/mixdata.JPG"
    return gr.Image(value=path)

def hint2_image(hint2_option):
    path = ""
    if hint2_option == "분류 힌트":
        path = "image/main/classific_task.JPG"
    elif hint2_option == "예측 힌트":
        path = "image/main/pred_task.JPG"
    return gr.Image(value=path)
def mv_stage_btn():
    return gr.Button(visible=True)
def next_q2():
    return gr.Text(visible=True), gr.Image(visible=True, elem_id='container'), gr.Radio(visible=True),gr.Dropdown(visible=True) ,gr.Text(visible=False), gr.Image(visible=False, elem_id=None),gr.Radio(visible=False), gr.Dropdown(visible=False)
def next_q3():
    return gr.Text(visible=False),gr.Text(visible=False),gr.Image(visible=False, elem_id=None), gr.Dropdown(visible=False), gr.Text(visible=True), gr.Image(visible=True, elem_id="container"), gr.Dropdown(visible=True)

            return classification_table.launch(inbrowser=True)

def next_interface(data_type, purpose, user_lev):
    if data_type == "시계열 데이터"and purpose == "예측":
        if user_lev == "없다":
            return ts_fore_lite.launch(inbrowser=True)
        elif user_lev == "없지만 직접하고 싶다" or user_lev == "경험이 있다":
            return ts_fore.launch(inbrowser=True)
    elif data_type =="테이블 데이터" and purpose == "예측":
        if user_lev == "없다":
            return regression_lite.launch(inbrowser=True)
        elif user_lev == "없지만 직접하고 싶다" or user_lev == "경험이 있다":
            return regression.launch(inbrowser=True)
with gr.Blocks(title="쉬운 머신 러닝?", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")),css="temp/style.css") as selection:
    gr.Markdown(
        """
        # 어떤 머신러닝 모델을 만들어드릴까요?
        """
        , elem_id='title'
    )

    with gr.Row(scale = 1):
        # 질의를 위한 column
        with gr.Column(scale=1):
            q1 = gr.Text(label="질문 1", value="데이터가 필요합니다 어떤 형태의 데이터를 가지고 계신가요?")
            r1 = gr.Dropdown(label="답변을 선택하세요", choices=["이미지 데이터", "테이블 데이터","시계열 데이터","혼합 데이터"],interactive=True)
            q2 = gr.Text(label="질문 2", value="어떤 목적으로 사용하려고 하나요?", visible=False)
            r2 = gr.Dropdown(label="답변을 선택하세요", choices=["예측하려고 한다.", "분류하려고 한다."], visible=False, interactive=True)
        with gr.Column(scale=1):
            hint1 = gr.Image(label="1번 질문 힌트", type="pil", container=True, elem_id='container')
            hint1_drop = gr.Radio(label="확인할 힌트를 선택하세요",value=None,choices=["이미지데이터", "테이블데이터","시계열데이터","혼합데이터"])
            hint2 = gr.Image(label="2번 질문 힌트", visible=False, value="image/main/classific_task.JPG", type="pil",
                                 container=True)
            hint2_drop = gr.Radio(label="확인할 힌트를 선택하세요",value=None,choices=["분류 힌트", "예측 힌트"], visible=False, interactive=True)
    with gr.Row(scale = 1):
        with gr.Column():
            q3 = gr.Text(label="질문 3", value="데이터 조작, 분석, 머신러닝과 관련한 경험이 있나요?", visible=False)
            hint3 = gr.Image(label="3번 질문 힌트", value="image/main/hint3.JPG", visible=False)
            r3 = gr.Dropdown(label="답변을 선택하세요", choices=["없다", "없지만 직접하고 싶다", "경험이 있다"], interactive=True, visible=False)

    stage1_btn = gr.Button(value="만들기 시작", visible=False)

    hint1_drop.change(fn=hint1_image, inputs=[hint1_drop], outputs=[hint1])
    r1.select(fn=next_q2, inputs=None, outputs=[q2, hint2, r2,hint2_drop ,q1, hint1, r1, hint1_drop])
    hint2_drop.change(fn=hint2_image, inputs=[hint2_drop], outputs=[hint2])
    r2.select(fn=next_q3, inputs=None, outputs=[q2, r2, hint2, hint2_drop, q3, hint3, r3])
    r3.select(fn = mv_stage_btn, inputs = None, outputs=[stage1_btn])
    # purpose.change(fn = next_q3, inputs=None, outputs=[q3,hint3 ,process_type,process_btn])
    stage1_btn.click(fn = next_interface, inputs=[r1, r2, r3], outputs=None)
if __name__ == "__main__":
    selection.launch(share=True)