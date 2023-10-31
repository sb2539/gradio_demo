import gradio as gr
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
import matplotlib.font_manager as fm
import matplotlib
import random

def name_to_rgb(color_name):
    rgb = mcolors.to_rgb(color_name)
    return tuple([int(255 * x) for x in rgb])  # RGB


def scale_value(value, old_min=8, old_max=4096, new_min=25, new_max=1024):
    return ((value - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min


def value_to_color(value, min=25, max=1024):
    # 값의 범위를 0~1 사이로 정규화
    normalized = (value - min) / max

    # 파란색 계열로 매핑 (RGB)
    blue = int((1 - normalized) * 255)
    return (255, 255, blue)


def add_rectangle(img, inner_color, border_color, position, size, text_scale, font_path, text):
    # 좌표와 크기 추출
    x, y = position
    width, height = size

    # 색상 이름을 RGB 값으로 변환
    inner_color = name_to_rgb(inner_color)
    border_color = name_to_rgb(border_color)

    # Draw 객체 생성
    draw = ImageDraw.Draw(img)

    # 사각형 그리기
    start_x = x + width // 2 - width // 2
    start_y = y
    end_x = start_x + width
    end_y = start_y + height
    draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=inner_color, outline=border_color)

    # 폰트 설정
    font = ImageFont.truetype(font_path, int(text_scale * 10))

    # 텍스트 위치 계산 (사각형 중앙)
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    text_x = start_x + width // 2 - text_width // 2
    text_y = start_y + height // 2 - text_height // 2

    # 텍스트 그리기
    draw.text((text_x, text_y), text, fill='black', font=font)

    return img


def add_rectangle2(img, inner_color, border_color, position, size, text_scale, font_path, text):
    # 좌표와 크기 추출
    x, y = position
    width, height = size

    # 색상 이름을 RGB 값으로 변환
    border_color = name_to_rgb(border_color)

    # Draw 객체 생성
    draw = ImageDraw.Draw(img)

    # 사각형 그리기
    start_x = x + width // 2 - width // 2
    start_y = y
    end_x = start_x + width
    end_y = start_y + height
    draw.rectangle([(start_x, start_y), (end_x, end_y)], fill=value_to_color(inner_color), outline=border_color)

    # # 폰트 설정
    font = ImageFont.truetype(font_path, int(text_scale * 10))

    # 텍스트 위치 계산 (사각형 중앙)
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    text_x = start_x + width // 2 - text_width // 2
    text_y = start_y + height // 2 - text_height // 2

    # 텍스트 그리기
    draw.text((text_x, text_y), text, fill='black', font=font)

    return img


def draw_arrow(img, rect_top_left, rect_size, direction, arrow_length, color='black', thickness=2):
    # 사각형의 위치와 크기 추출
    (x1, y1) = rect_top_left
    (width1, height1) = rect_size

    # 화살표의 시작점 설정 (사각형의 변 중심)
    if direction == 'up':
        start = (x1 + width1 // 2, y1)
        end = (x1 + width1 // 2, y1 - arrow_length)
    elif direction == 'down':
        start = (x1 + width1 // 2, y1 + height1)
        end = (x1 + width1 // 2, y1 + height1 + arrow_length)
    elif direction == 'left':
        start = (x1, y1 + height1 // 2)
        end = (x1 - arrow_length, y1 + height1 // 2)
    else:  # 'right'
        start = (x1 + width1, y1 + height1 // 2)
        end = (x1 + width1 + arrow_length, y1 + height1 // 2)

    # 화살표 그리기
    draw = ImageDraw.Draw(img)
    draw.line([start, end], fill=color, width=thickness)

    # 화살표 머리 그리기
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_head_start = (end[0] - arrow_length * math.cos(angle), end[1] - arrow_length * math.sin(angle))
    arrow_head_left = (arrow_head_start[0] + arrow_length * math.cos(angle - math.pi / 4),
                       arrow_head_start[1] + arrow_length * math.sin(angle - math.pi / 4))
    arrow_head_right = (arrow_head_start[0] + arrow_length * math.cos(angle + math.pi / 4),
                        arrow_head_start[1] + arrow_length * math.sin(angle + math.pi / 4))
    draw.polygon([end, arrow_head_left, arrow_head_right], fill=color)

    return img

def remove_non_conv2d(module):
    for name, sub_module in list(module.named_children()):
        if isinstance(sub_module, nn.Conv2d):
            continue
        elif isinstance(sub_module, nn.Module):
            if len(list(sub_module.children())) == 0:
                delattr(module, name)
            else:
                remove_non_conv2d(sub_module)

def visualize_yolov2(model):
    # 모델(YOLOv2) 호출
    # model = YOLOv2Tiny(20, [(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)])
    # if str(type(model))=="<class 'tempfile._TemporaryFileWrapper'>":
    #     model = model.name
    model = torch.load(model)
    font_path = 'temp/Jua-Regular.ttf'
    scale_factor = 2
    background_X = 1200
    background_Y = 1300

    # 배경 생성
    img = Image.new('RGB', (background_X * scale_factor, background_Y * scale_factor), 'white')
    x_position = 100
    y_position = 150
    x_size = 150
    y_size = 60
    y_change = 80
    font_size = 3
    #### 모델 구조를 그리는 함수 ####
    layers = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3', 'conv4', 'pool4', 'conv5', 'pool5', 'conv6',
              'pool6', 'conv7', 'conv8_1', 'conv8_2']

    for i, layer_name in enumerate(layers):
        if 'conv' in layer_name:
            color = 'lightsalmon'
        else:
            color = 'lightskyblue'
        img = add_rectangle(  # 왼쪽 모델 구조
            img=img,
            inner_color=color,
            border_color='black',
            position=(x_position, y_position),
            size=(x_size * scale_factor, y_size * scale_factor),
            text_scale=font_size * scale_factor,
            font_path=font_path,
            text=layer_name)
        if i != len(layers) - 1:
            draw_arrow(img, (x_position, y_position), (x_size * scale_factor, y_size * scale_factor), direction='down',
                       arrow_length=(y_change - y_size) * scale_factor, color='black', thickness=2 * scale_factor)
        # 모델 구조의 사각형 아래에 화살표 생성, 마지막인 경우 생성 x

        if layer_name == 'conv8_2':  # Conv말고 nn의 Conv2d쓰는 얘
            out_channels = getattr(model, layer_name).out_channels
            # 사각형의 크기 계산 (입력 채널 수를 너비로 사용)
            width = int(scale_value(out_channels) * scale_factor)

            # 사각형 추가
            img = add_rectangle2(
                img=img,
                inner_color=width,
                border_color='black',
                position=(x_position + x_size * scale_factor + 25 * scale_factor, y_position),
                size=(width, y_size * scale_factor),
                text_scale=font_size * scale_factor,
                font_path=font_path,
                text=str(out_channels)
            )
        elif 'conv' in layer_name:  # 오른쪽 모델 레이어 크기, 채널 개수로 길이 지정
            out_channels = getattr(model, layer_name).conv.out_channels
            # 사각형의 크기 계산 (입력 채널 수를 너비로 사용)
            width = int(scale_value(out_channels) * scale_factor)
            # 사각형 추가
            img = add_rectangle2(
                img=img,
                inner_color=width,
                border_color='black',
                position=(x_position + x_size * scale_factor + 25 * scale_factor, y_position),
                size=(width, y_size * scale_factor),
                text_scale=font_size * scale_factor,
                font_path=font_path,
                text=str(out_channels)
            )
        else:  # maxpooling같은 것
            pass

        y_position += y_change * scale_factor

    enhancer = ImageEnhance.Sharpness(img)
    shapness_factor = 1
    img = enhancer.enhance(shapness_factor)

    # # 이미지 출력하기 위해 matplotlib 사용
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()
    # img.save('output.png')

    # 모델 크기 계산
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    #
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #
    # model_size = (param_size + buffer_size) / (1024 ** 2)
    #
    # print(f'model size: {model_size:.3f}MB')

    return img
def visualize_alexnet(model):
    # if str(type(model))=="<class 'tempfile._TemporaryFileWrapper'>":
    #     model = model.name
    # model = torch.load(model)
    font_path = 'temp/Jua-Regular.ttf'
    scale_factor = 2
    background_X = 1200
    background_Y = 1300

    # 배경 생성
    img = Image.new('RGB', (background_X * scale_factor, background_Y * scale_factor), 'white')
    x_position = 100
    y_position = 150
    x_size = 150
    y_size = 60
    y_change = 80
    font_size = 3
    #### 모델 구조를 그리는 함수 ####
    # 출력 채널 개수를 저장할 리스트를 초기화합니다.
    output_channels = []
    conv_indices = 0
    linear_indices = 0
    # 모델의 모든 레이어를 반복합니다.

    for name, layer in model.named_modules():
        # 만약 레이어가 컨볼루션 레이어라면,
        if isinstance(layer, torch.nn.Conv2d):
            conv_indices += 1
            # 레이어 이름과 출력 채널의 개수를 저장합니다.
            output_channels.append(("Conv" + str(conv_indices), layer.out_channels))
        # 만약 레이어가 선형 레이어라면,
        elif isinstance(layer, torch.nn.Linear):
            linear_indices += 1
            # 레이어 이름과 출력 채널의 개수를 저장합니다.
            output_channels.append(("Linear" + str(linear_indices), layer.out_features))
        elif isinstance(layer, torch.nn.MaxPool2d):
            output_channels.append(("MaxPool", 0))
        elif isinstance(layer, torch.nn.AdaptiveAvgPool2d):
            output_channels.append(("AdaptiveAvgPool", 0))
    print(output_channels)

    for i, (layer_name, num_channel) in enumerate(output_channels):
        if 'Conv' in layer_name:
            color = 'lightsalmon'
        elif 'MaxPool' in layer_name:
            color = 'seagreen'
        elif 'AdaptiveAvgPool' in layer_name:
            color = 'slategrey'
        else:
            color = 'lightskyblue'
        img = add_rectangle(  # 왼쪽 모델 구조
            img=img,
            inner_color=color,
            border_color='black',
            position=(x_position, y_position),
            size=(x_size * scale_factor, y_size * scale_factor),
            text_scale=font_size * scale_factor,
            font_path=font_path,
            text=layer_name)
        if i != len(output_channels) - 1:
            draw_arrow(img, (x_position, y_position), (x_size * scale_factor, y_size * scale_factor), direction='down',
                       arrow_length=(y_change - y_size) * scale_factor, color='black', thickness=2 * scale_factor)
        # 모델 구조의 사각형 아래에 화살표 생성, 마지막인 경우 생성 x

        # 사각형의 크기 계산 (입력 채널 수를 너비로 사용)
        width = int(scale_value(num_channel) * scale_factor)
        # 사각형 추가
        if num_channel==0:
            pass
        else:
            img = add_rectangle2(
                img=img,
                inner_color=width,
                border_color='black',
                position=(x_position + x_size * scale_factor + 25 * scale_factor, y_position),
                size=(width, y_size * scale_factor),
                text_scale=font_size * scale_factor,
                font_path=font_path,
                text=str(num_channel)
            )
        # else:  # maxpooling같은 것
        #     pass

        y_position += y_change * scale_factor

    enhancer = ImageEnhance.Sharpness(img)
    shapness_factor = 1
    img = enhancer.enhance(shapness_factor)

    # # 이미지 출력하기 위해 matplotlib 사용
    # plt.axis("off")
    # plt.imshow(img)
    # plt.show()
    # img.save('output.png')

    # 모델 크기 계산
    # param_size = 0
    # for param in model.parameters():
    #     param_size += param.nelement() * param.element_size()
    #
    # buffer_size = 0
    # for buffer in model.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #
    # model_size = (param_size + buffer_size) / (1024 ** 2)
    #
    # print(f'model size: {model_size:.3f}MB')

    return img
def visualize(file):
    if str(type(file))=="<class 'tempfile._TemporaryFileWrapper'>":
        df = pd.read_csv(file.name)
    df = pd.read_csv(file.name)
    return df
    # model = torch.load(model)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
    # # print(model.__class__.__name__)
    # if model.__class__.__name__ == "YOLOv2Tiny":
    #     return visualize_yolov2(model)
    # elif model.__class__.__name__ == "AlexNet":
    #     return visualize_alexnet(model)
    # else:
    #     print("그런 모델 지원 안해!")
def metrics(file_path): #TODO: Latency와 GPU, Model size는 적을수록 그래프 높게 나타내기
    df = pd.read_csv(file_path)
    data = df[['Accuarcy', 'Latency', 'Throughput', 'GPU_memory_footprint', 'Model_size']]
    categories = data.columns.values
    for col in data.columns:
        data[col] = (data[col] / data[col].max()) * 100

    # for col in data.columns:
    #     if col in ['Accuarcy', 'Throughput']:
    #         data[col] = (data[col] / data[col].max()) * 100
    #     else: # For 'Latency', 'GPU_memory_footprint', 'Model_size'
    #         data[col] = (1 - data[col] / data[col].max()) * 100
    fig = go.Figure()
    for i in range(len(df)):
        fig.add_trace(go.Scatterpolar(
            r=[data.loc[i, 'Accuarcy'], data.loc[i, 'Throughput'], data.loc[i, 'Latency'], data.loc[i, 'GPU_memory_footprint'], data.loc[i, 'Model_size']],
            customdata=[str(df.loc[i, 'Accuarcy'])+' %', str(df.loc[i, 'Throughput'])+' fps', str(df.loc[i, 'Latency'])+' ms',
                        str(df.loc[i, 'GPU_memory_footprint'])+' MB', str(df.loc[i, 'Model_size'])+' MB'],
            theta=categories,
            hovertemplate=
            '<b>%{theta}</b>' +
            '<br>Normalized Value: %{r}' +
            '<br>Original Value: %{customdata}',
            fill='toself',
            name=df.loc[i, 'Model_structure']
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                ticksuffix='%',
                range=[0,100],
            )),
        hovermode='closest',
        font_family='Jua',
        title=dict(
            text='Matrics',
            x=0.5,
            font=dict(size=40)),
        title_font_family="Jua",
        showlegend=True,
        legend=dict(
            x=0,
            y=0.5,
            traceorder="normal",
            font=dict(
                family="Jua, Nanum Gothic Coding",
                size=12,
                color="slateblue"
            ),
            bgcolor="lavender",
            bordercolor="Black",
            borderwidth=2
        )
    )
    return fig
# def upload_file_to_dataframe(input_file):
#     df = pd.read_csv(input_file.name)
#     return df
def upload_file_to_dataframe(file):
    if file is not None:
        df = pd.read_csv(file.name)  # 업로드된 파일을 DataFrame으로 변환
        return df
    else:
        return None

def make_plot(plot_type,data):
    if plot_type == '없는값':
        fig = plt.figure()
        nan_counts = data.isna().sum()
        nan_counts.plot(kind='bar')
        return fig
    elif plot_type == '0인값':
        fig = plt.figure()
        zero_counts = data.eq(0).sum()
        zero_counts.plot(kind='bar')
        return fig
    # elif plot_type == '관련성': # 이 부분 수정하기
    #     fig = plt.figure()
    #     sns.heatmap(data.corr())
    #     plt.title("heatmap")
    #     plt.ylabel("GDP (Millions)")
    #     plt.xlabel("Population Change since 1800")
    #     return fig

def plot_after_modi(plot_type, data, ax_x, ax_y):
    if plot_type == '상관관계':
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(),annot=True)
        plt.title("히트맵")
        return fig
    elif plot_type == '다른 그래프':
        fig = plt.figure()
        sns.barplot(x = ax_x, y=ax_y, hue=ax_x ,data = data, palette="BuPu", legend=False)
        plt.title("평균 일조량")
        return fig
def data_drop(check_col, data):
    droped_data = data.drop(columns=check_col)
    return droped_data

def modi_nan(nan_radio, data):
    if nan_radio == "제거":
        data = data.dropna()
        return data
    elif nan_radio == "채우기":
        data = data.fillna(data.mean())
        return data
    else :
        return data
def get_col(data):
    column_list = []
    for column in data.columns:
        column_list.append(column)
    return column_list

with gr.Blocks(title="일조량 예측 demo", theme=gr.themes.Soft(font=fonts.GoogleFont("Jua")), css="temp/style.css") as demo:
    gr.Markdown(
        """
        # 일조량 예측
        설명 넣기?
        """
        , elem_id='title'
    )
    with gr.Row():
        with gr.Column(scale=1):
        # phase 1: 데이터 입력 파트
            task_type = gr.Radio(["예측하기", "분류하기"], value="예측하기", label="업무 선택")
            upload_button = gr.UploadButton(label="데이터 업로드", file_types=['.pth', '.pt', '.tar', '.csv'], file_count="single",
                                            variant="primary")
            df_for_plot = gr.DataFrame(visible=False) # 0인값, nan 인값 볼려고 만든 테이블
            #data_info = gr.DataFrame(label="데이터 정보", visible=False)

        #데이터 분석 파트
            data_des = gr.DataFrame(label = "데이터 분석", visible=False)
            button = gr.Radio(label='그래프 선택', choices=['없는값', '0인값', '그래프 안보기'],visible=False, value='그래프 안보기')
            show_graph = gr.Plot(visible=False, label='그래프')
            #show_graph = gr.Gallery(visible=False, label='그래프')
            button.change(make_plot, inputs=[button, df_for_plot], outputs=show_graph)
        #데이터 수정 파트
            df_maketime_fet = gr.DataFrame(visible=False)
            df_column_check = gr.CheckboxGroup(visible=False, label='데이터 열')
            delete_btn = gr.Button(visible=False, value='열 삭제')
            delete_btn.click(fn =data_drop, inputs=[df_column_check, df_maketime_fet], outputs=df_maketime_fet)
            x_axis = gr.Dropdown(visible=False, label="x축")
            y_axis = gr.Dropdown(visible=False, label="Y축")



        with gr.Column(scale=1):
            data_frame = gr.DataFrame(label="일조량 데이터") # 이후 원본 데이터 출력 칸
        # 데이터 분석 파트
            nan_radio = gr.Radio(label='없는 값 처리', choices=['제거', '채우기', '필요 없음'], visible=False, value='필요 없음',interactive=True)
            #choose_nan_col = gr.Dropdown(label='없는 값 채워야 하는 열', visible=False)
            nan_btn = gr.Button(visible=False, value='수정')
            nan_btn.click(fn = modi_nan, inputs=[nan_radio, data_frame], outputs=[data_frame])
            nan_btn.click(fn=modi_nan, inputs=[nan_radio, data_frame], outputs=[df_for_plot])
        # 데이터 수정 파트
            after_modi_radio = gr.Radio(label='그래프 선택', choices=['상관관계', '다른 그래프', '그래프 안보기'], visible=False, value='그래프 안보기')
            modi_graph = gr.Plot(visible=False, label='그래프', scale=5, min_width=500)
            after_modi_radio.change(fn = plot_after_modi, inputs = [after_modi_radio, df_maketime_fet, x_axis, y_axis], outputs=modi_graph)
        # phase 1
            #model_structure = gr.Image(height=600, type='filepath',label="모델 파일을 입력하면 모델 구조를 시각화 합니다.") #TODO 초기 화면에 튜토리얼 이미지 추가
        # phase 2
            #imagebox = gr.Image(height=600, visible=False, label="검색된 모델 구조")

    preprocess_btn = gr.Button("데이터 분석", variant="primary")
    data_modify_btn = gr.Button(value="데이터 수정하기", visible=False)

    upload_button.upload(upload_file_to_dataframe,inputs=upload_button, outputs=data_frame)
    #upload_button.upload(visualize, upload_button, data_frame)
    # phase 2: 검색 완료 후 다음 페이즈 진행 버튼
    finish_btn = gr.Button("검색 결과 보기", visible=False)

    # phase 3: 검색 종료 후 결과
    matrix_plot = gr.Plot(label="레이블 클릭 시 ON/OFF", visible=False)
    download_btn = gr.Button("후보 모델 다운로드", visible=False) # TODO: 후보 모델들을 저장할 수 있도록 하기
    download_link = gr.File(visible=False)
    return_btn = gr.Button("처음으로 돌아가기", visible=False)

    # def datadescribe(dataframe):
    #     data_des = data_frame.describe()
    #     return data_des
    @preprocess_btn.click(inputs=[data_frame],
                          outputs=[task_type, upload_button,data_frame ,data_des,
                                   df_for_plot, button, show_graph,preprocess_btn,data_modify_btn,nan_radio,nan_btn])
    def data_info(datframe):
        #data_information = data_frame.info()
        data_describe = datframe.describe()
        data_describe = data_describe.reset_index()
        column_list = []
        for column in datframe.columns:
            column_list.append(column)
        return {
            task_type: gr.Radio(visible=False),
            upload_button : gr.UploadButton(visible=False),
            #data_frame : gr.update(label='원본데이터',visible=True),
            data_frame: gr.DataFrame(label='원본 데이터', visible=True),
            data_des : gr.DataFrame(value = data_describe, visible=True),
            df_for_plot : gr.DataFrame(value=datframe, visible=False),
            button : gr.Radio(visible=True),
            show_graph : gr.Plot(visible=True),
            preprocess_btn:gr.Button(visible=False),
            data_modify_btn : gr.Button(visible = True),
            nan_radio : gr.Radio(visible=True),
            # choose_nan_col : gr.Dropdown(visible=True, choices=column_list, value=lambda : random.choice(column_list),interactive=True, allow_custom_value=True),
            nan_btn : gr.Button(visible=True)
            #show_graph : gr.Gallery(visible=True)
        }

    def make_time_fet(data):
        time_columns = []
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



    @data_modify_btn.click(inputs=[data_frame],
                          outputs=[task_type, upload_button, data_frame, data_des,
                                   df_for_plot, button, show_graph, preprocess_btn, data_modify_btn, nan_radio,nan_btn
                                   ,df_maketime_fet,df_column_check,delete_btn,x_axis,y_axis,after_modi_radio,modi_graph])
    def data_modify(datframe):
        modified_data = make_time_fet(datframe)
        column_list = []
        for column in modified_data.columns:
            column_list.append(column)
        return{task_type: gr.Radio(visible=False),
            upload_button : gr.UploadButton(visible=False),
            #data_frame : gr.update(label='원본데이터',visible=True),
            data_frame: gr.DataFrame(label='원본 데이터', visible=True),
            data_des : gr.DataFrame(visible=False),
            df_for_plot : gr.DataFrame(visible=False),
            button : gr.Radio(visible=False),
            show_graph : gr.Plot(visible=False),
            preprocess_btn:gr.Button(visible=False),
            data_modify_btn : gr.Button(visible = False),
            nan_radio : gr.Radio(visible=False),
            # choose_nan_col : gr.Dropdown(visible=False),
            nan_btn : gr.Button(visible=False),
            df_maketime_fet : gr.DataFrame(value=modified_data, visible=True,label='시간정보 추가 데이터'),
            df_column_check : gr.CheckboxGroup(choices=column_list, visible=True, interactive=True, value = 'Data', info="삭제할 열을 선택하세요"),
            delete_btn : gr.Button(visible=True),
            x_axis : gr.Dropdown(visible=True, interactive=True, choices=column_list),
            y_axis: gr.Dropdown(visible=True, interactive=True,choices=column_list),
            after_modi_radio : gr.Radio(visible=True),
            modi_graph : gr.Plot(visible=True)
               }

    # @preprocess_btn.click(inputs=[dataset, task_type, acc, latecny], outputs=[task_type, model_structure, preprocess_btn, upload_button, acc, latecny, dataset, graph, imagebox, finish_btn])
    # def search_arch(benchmark_dataset, model_type_value, acc_value, latecny_value):
    #     ########################################################################
    #     #TODO: 여기서 NAS가 실행되면서 model structure와 model metrics를 계산하면 됨#
    #     #      inputs와 입력 변수에 필요한 것들 추가하면 됨                         #
    #     ########################################################################
    #     model_paths = []
    #     models_path = Path('temp/model')
    #     files_in_basepath = (entry for entry in models_path.iterdir() if entry.is_file())
    #     matrics = pd.read_csv(r'C:\Users\Pc\Desktop\visualization\temp\history.csv')
    #     name = matrics["Iteration"]
    #     x = matrics["Resource"]
    #     y = matrics["Accuracy"]
    #     z = matrics["Modelsize"]
    #     metrics_plot = go.Figure(go.Scatter(x=x[:1], y=y[:1], mode='markers',
    #                                         marker=dict(size=z[:1]+10, sizemode='diameter', color='LightSkyBlue'),
    #                                         hovertemplate=
    #                                         f'{name.values[0]}' +
    #                                         '<br>Layency: %{x:.4f} ms' +
    #                                         '<br>Accuracy: %{y:.2f} %'+
    #                                         f'<br>Model size: {z[0]} MB'+
    #                                         '<extra></extra>',
    #                                         showlegend=False))
    #     # 초기 화면의 제약조건에 따라 빨간 점선 그려줌
    #     metrics_plot.add_vline(x=latecny_value, line_width=2, line_dash="dash", line_color="red")
    #     metrics_plot.add_hline(y=acc_value, line_width=2, line_dash="dash", line_color="red")
    #
    #     # 플롯 영역의 배경을 흰색으로 설정
    #     # x, y 범위 지정
    #     metrics_plot.update_xaxes(range=[0.01, 0.05], showline=True,)
    #     metrics_plot.update_yaxes(range=[25, 40], showline=True,)
    #
    #     metrics_plot.update_layout(
    #         # 배경 하얀색
    #         paper_bgcolor="white", plot_bgcolor='white',
    #         # 축 추가
    #         xaxis=dict(
    #             showline=True,
    #             linewidth=2,
    #             linecolor='black',
    #             mirror=False
    #         ),
    #         yaxis=dict(
    #             showline=True,
    #             linewidth=2,
    #             linecolor='black',
    #             mirror=False
    #         ),
    #         # 축이름
    #         xaxis_title="Latency (ms)",
    #         yaxis_title="Accuracy (%)",
    #         # 폰트
    #         font=dict(
    #             family='Jua',
    #             size=16,
    #             color="RebeccaPurple"
    #         )
    #     )
    #     for item in files_in_basepath:
    #         model_paths.append(item.name)
    #     i=0
    #     twinkle=7 # 홀수로 맞출 것
    #     twinkle_check=0
    #     for _ in range(len(model_paths)+twinkle):
    #         model_path=os.path.join(os.getcwd()+'\\temp\\model',model_paths[i])
    #         if model_type_value=="YOLOv2 tiny":
    #             model_graph = visualize_yolov2(model_path)
    #         elif model_type_value=="YOLOv5n":
    #             model_graph = visualize_alexnet(model_path)
    #         else:
    #             model_graph = None
    #         # from time import sleep
    #         # sleep(0.2)
    #
    #         if i == 0:
    #             i += 1
    #             yield {
    #                 task_type: gr.update(visible=False),
    #                 model_structure: gr.update(visible=False),
    #                 preprocess_btn: gr.update(visible=False),
    #                 upload_button: gr.update(visible=False),
    #                 acc: gr.update(visible=False),
    #                 latecny: gr.update(visible=False),
    #                 dataset: gr.update(visible=False),
    #                 graph: gr.update(value=metrics_plot, visible=True),
    #                 imagebox: gr.update(value=model_graph, visible=True),
    #             }
    #         elif i == len(model_paths)-1:
    #             if twinkle_check%2==0:
    #                 metrics_plot.add_trace(go.Scatter(x=x[-1:], y=y[-1:], mode='markers',
    #                                                   marker=dict(size=z[-1:]+10, sizemode='diameter',color='white'),
    #                                                   hovertemplate=
    #                                                   f'{name.values[len(model_paths)-1]}' +
    #                                                   '<br>Layency: %{x:.4f} ms' +
    #                                                   '<br>Accuracy: %{y:.2f} %'+
    #                                                   f'<br>Model size: {z[len(model_paths)-1]} MB'+
    #                                                   '<extra></extra>',
    #                                                   showlegend=False))
    #             else:
    #                 metrics_plot.add_trace(
    #                     go.Scatter(x=x[-1:], y=y[-1:], mode='markers', marker=dict(size=z[-1:]+10, sizemode='diameter',color='red'), hovertemplate=
    #                                         f'{name.values[len(model_paths)-1]}' +
    #                                         '<br>Layency: %{x:.4f} ms' +
    #                                         '<br>Accuracy: %{y:.2f} %'+
    #                                         f'<br>Model size: {z[len(model_paths)-1]} MB'+
    #                                         '<extra></extra>',
    #                                         showlegend=False))
    #             if twinkle_check == twinkle:
    #                 yield {
    #                     graph: gr.update(value=metrics_plot, visible=True),
    #                     imagebox: gr.update(value=model_graph, visible=True),
    #                     finish_btn: gr.update(visible=True) # 마지막에 결과 버튼 나타내기
    #                 }
    #             else:
    #                 twinkle_check += 1
    #                 yield {
    #                     graph: gr.update(value=metrics_plot, visible=True),
    #                     imagebox: gr.update(value=model_graph, visible=True),
    #                 }
    #         else:
    #             metrics_plot.add_trace(go.Scatter(x=x[i:i+1], y=y[i:i+1], mode='markers',
    #                                               marker=dict(size=z[i:i+1]+10, sizemode='diameter',color='black'),
    #                                               hovertemplate=
    #                                               f'{name.values[i]}' +
    #                                               '<br>Layency: %{x:.4f} ms' +
    #                                               '<br>Accuracy: %{y:.2f} %'+
    #                                               f'<br>Model size: {z[i]} MB'+
    #                                               '<extra></extra>',
    #                                               showlegend=False))
    #             i+=1
    #             yield {
    #                 graph: gr.update(value=metrics_plot, visible=True),
    #                 imagebox: gr.update(value=model_graph, visible=True),
    #             }

    # @finish_btn.click(inputs=None, outputs=[graph, imagebox, finish_btn, matrix_plot, return_btn, download_btn])
    # def show_result():
    #     matrix = metrics(r'temp\data.csv')
    #     return {
    #         graph: gr.update(visible=False),
    #         imagebox: gr.update(visible=False),
    #         finish_btn: gr.update(visible=False),
    #         matrix_plot: gr.update(value=matrix, visible=True),
    #         return_btn: gr.update(visible=True),
    #         download_btn: gr.update(visible=True)
    #     }
    # @download_btn.click(inputs=None, outputs=[download_btn,download_link])
    # def download_file():
    #     #TODO: 후보 모델'들' 다운로드할 수 있도록 하기 : 압축파일로 만들어서 보내거나 해야될 듯
    #     return {
    #         download_btn:gr.update(visible=False),
    #         download_link:gr.update(value=r"C:\Users\Pc\Desktop\visualization\temp\model\iter_30_best_model.pth.tar", visible=True)
    #     }
    #
    # @return_btn.click(inputs=None, outputs=[model_structure, task_type, preprocess_btn, upload_button, acc, latecny, dataset, graph, imagebox, matrix_plot, return_btn, download_btn, download_link])
    # def home():
    #     return {
    #         model_structure: gr.update(value=None, visible=True),
    #         task_type: gr.update(visible=True),
    #         preprocess_btn: gr.update(visible=True),
    #         upload_button: gr.update(visible=True),
    #         acc: gr.update(visible=True),
    #         latecny: gr.update(visible=True),
    #         dataset: gr.update(visible=True),
    #         graph: gr.update(value=None, visible=False),
    #         imagebox: gr.update(value=None, visible=False),
    #         matrix_plot: gr.update(value=None, visible=False),
    #         return_btn: gr.update(visible=False),
    #         download_btn: gr.update(visible=False),
    #         download_link: gr.update(visible=False)
    #     }
if __name__ == "__main__":
    font_path = r"C:\Users\sinb1\Downloads\nanum-all\namunamu\namu\NanumFontSetup_TTF_BARUNGOTHIC\NanumBarunGothic.ttf"
    font_prop = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = font_prop
    plt.rcParams['axes.unicode_minus'] = False
    print(matplotlib.get_cachedir())
    demo.queue().launch()
    # demo.queue().launch(share=True)