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


# 분류 인터페이스
#classification_interface = gr.Interface(fn=classify_or_regress, inputs=gr.Radio(visible=True, choices=["fuck"]), outputs=gr.Text(value="bye"))
with gr.Blocks(title="분류 모델을 만듭니다.", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")), css="temp/style.css") as classification:
    gr.Markdown(
        """
        # 지금부터 분류 모델 만드는 도움을 드릴게요
        """
        , elem_id='title'
    )
    with gr.Column(scale=1):
        # phase 1: 데이터 입력 파트
        task_type = gr.Radio(["예측하기", "분류하기"], value="예측하기", label="업무 선택")
        upload_button = gr.UploadButton(label="데이터 업로드", file_types=['.pth', '.pt', '.tar', '.csv'],
                                        file_count="single",
                                        variant="primary")
        df_for_plot = gr.DataFrame(visible=False)