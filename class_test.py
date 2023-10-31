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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from scipy import stats
import copy
from sklearn.model_selection import train_test_split
from automl.automl import pycaret_w, h2o_w
from other_task.make_classification_table import classification_table
from other_task.make_regressor import regression
from other_task.make_classification import classification
from other_task.make_nontime_regressor import non_time_regression
from automl.automl import AutoMl_library

dataurl = "C:\\Users\\sin\\PycharmProjects\\sibal\\data\\SolarPrediction.csv"

class Dataload_anl():
    def __init__(self, dataurl ,drop_option, target):
        self.url = dataurl
        self.drop_option = drop_option
        self.target_col = target

    def data_to_csv(self):
        df = pd.read_csv(self.url)
        return df
    def get_target(self, data):
        target_data = data[self.target_col]
        target_data = pd.DataFrame(target_data)
        return target_data
    def drop_nan(self, data):
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            if self.drop_option == "삭제":
                data = data.dropna()
            elif self.drop_option == "채우기":
                data = data.fillna(data.mean())
        return data

    def data_transformation(self, data):
        for column in data.columns:
            if re.search(r'\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}', str(data[column])):
                data[column] = data[column].apply(lambda x: x.split()[0])
                data['Month'] = pd.to_datetime(data[column]).dt.month
                data['Day'] = pd.to_datetime(data[column]).dt.day
            elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column])) and len(column) < 5:
                # data[column] = data[column].apply(lambda x: x.split()[0])
                data['Hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
                data['Minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute
                data['Second'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.second

            elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column])) and 'Rise' in column:
                data['Rise_hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
                data['Rise_minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute

            elif re.search(r'\d{2}:\d{2}:\d{2}', str(data[column])) and 'Set' in column:
                data['Set_hour'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.hour
                data['Set_minute'] = pd.to_datetime(data[column], format='%H:%M:%S').dt.minute

        return data

class Drop_data():
    def __init__(self, choosed_column):
        self.choosed_column = choosed_column

    def drop_in_column(self, data):
        droped_data = data.drop(columns=self.choosed_column)
        return droped_data

class Data_preprocess():
    def __init__(self, trans_type, split_count, target):
        self.type = trans_type
        self.split_count = split_count
        self.target_col = target
    def data_split(self, data, target_data):
        copy_data = copy.deepcopy(data) # automl 라이브러리 중 일부는 target이 포함된 데이터가 필요함
        input_feature = copy.deepcopy(data)
        input_feature = input_feature.drop(self.target_col, axis = 1)
        xtrain, xtest, ytrain, ytest = train_test_split(input_feature, target_data, test_size=self.split_count, random_state=1)
        ### 'Standard','MinMax', "MaxAbsScaler", RobustScaler
        if self.type == "Standard":
            scaler = StandardScaler()
            xtrain = scaler.fit_transform(xtrain)
            xtest = scaler.transform(xtest)
        elif self.type == "MaxAbsScaler":
            scaler = MaxAbsScaler()
            xtrain = scaler.fit_transform(xtrain)
            xtest = scaler.transform(xtest)
        elif self.type == "RobustScaler":
            scaler = RobustScaler()
            xtrain = scaler.fit_transform(xtrain)
            xtest = scaler.transform(xtest)
        elif self.type == "MinMax":
            scaler = MinMaxScaler()
            xtrain = scaler.fit_transform(xtrain)
            xtest = scaler.transform(xtest)
        return xtrain, xtest, ytrain, ytest, copy_data




if __name__ == "__main__":
    target = "Radiation"
    data_load_anlysis = Dataload_anl("C:\\Users\\sin\\PycharmProjects\\sibal\\data\\SolarPrediction.csv", "삭제", target)
    df = data_load_anlysis.data_to_csv()
    print(df)
    target_data = data_load_anlysis.get_target(df)
    print(target_data)
    df = data_load_anlysis.drop_nan(df)
    print(df)
    df = data_load_anlysis.data_transformation(df)
    print(df)
    column_list = ["UNIXTime", "Data", "Time", "TimeSunRise","TimeSunSet"]
    data_drop = Drop_data(column_list)
    df = data_drop.drop_in_column(df)
    print(df)
    trans_type = "Standard"
    split = 0.2
    data_process = Data_preprocess(trans_type, split, target) # 데이터 전처리
    x_train, x_test, y_train, y_test, copy_data = data_process.data_split(df, target_data)
    print(x_train.shape)
    print(x_test.shape)
    automl_library = AutoMl_library(copy_data, x_train, x_test, y_train, y_test, split, "Regressor", target)
    pycaret = automl_library.pycaret_w(copy_data)
    h2o_auto = automl_library.h2o_w(copy_data)
    print(pycaret)
    print(h2o_auto)