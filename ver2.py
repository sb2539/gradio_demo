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
from other_task.make_classification_table import classification_table
from other_task.make_regressor import regression
from other_task.make_classification import classification
from other_task.make_nontime_regressor import non_time_regression


def next_q2():
    return gr.Text(visible=True), gr.Image(visible=True), gr.Dropdown(visible=True)
def next_q3():
    return gr.Text(visible=True), gr.Radio(visible=True), gr.Button(visible=True)
def next_interface(data_type, purpose, process_type):
    if data_type == "표":
        if purpose == "표 내부에 있는 정보를 예측하고 싶다." and process_type == "아니오":
            return regression.launch(inbrowser=True)
        elif purpose == "표 내부에 있는 정보를 예측하고 싶다." and process_type == "예":
            return regression.launch(inbrowser=True)
        elif purpose == "표 내부의 정보를 판별하고 싶다." and process_type == "예":
            return classification_table.launch(inbrowser=True)
        elif purpose == "표 내부의 정보를 판별하고 싶다." and process_type == "아니오":
            return classification_table.launch(inbrowser=True)
    elif data_type == "이미지":
        if purpose == "이미지를 판별하고 싶다." and process_type == "아니오":
            return classification.launch(inbrowser=True)
        elif purpose == "이미지를 판별하고 싶다." and process_type == "예":
            return classification.launch(inbrowser=True)

with gr.Blocks(title="어떤 모델을 만들어줄까요?", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")),css="temp/style.css") as selection:
    gr.Markdown(
        """
        # 어떤 머신러닝 모델을 만들어드릴까요?
        """
        , elem_id='title'
    )
    # 첫번째 질문
    q1 = gr.Text(label="질문 1", value="데이터가 필요합니다 어떤 형태의 데이터를 가지고 계신가요?")
    hint1 = gr.Image(label="답변이 어렵다면 참고해주세요", value="image/dataex.PNG", type="pil", container=False)
    job = gr.Dropdown(label="답변을 선택하세요", choices=["이미지", "표"])

    # 두번째 질문
    q2 = gr.Text(label="질문 2", value="데이터로 하고자 하는 일이 무엇인가요?", visible=False)
    hint2 = gr.Image(label="답변이 어렵다면 참고해주세요", visible=False,value="image/taskex.PNG", type="pil", container=False)
    purpose = gr.Dropdown(label="관련 있는 답변을 선택하세요", choices=["이미지를 판별하고 싶다.", "표 내부에 있는 정보를 예측하고 싶다.", "표 내부의 정보를 판별하고 싶다."], visible=False)

    # 세번째 질문
    q3 = gr.Text(label="질문 3", value="이후의 과정을 기본 값으로 진행할까요? 아니오를 선택하시면 사용자의 선택에 따라 진행됩니다.", visible=False)
    process_type = gr.Radio(label="선택해주세요", choices=["예", "아니오"], visible=False)
    process_btn = gr.Button(value="선택하신 답변을 토대로 실행됩니다.", visible=False)

    job.change(fn=next_q2, inputs=None, outputs=[q2, hint2, purpose])
    purpose.change(fn = next_q3, inputs=None, outputs=[q3, process_type,process_btn])
    process_btn.click(fn = next_interface, inputs=[job, purpose, process_type], outputs=None)
if __name__ == "__main__":
    selection.launch(share=True)