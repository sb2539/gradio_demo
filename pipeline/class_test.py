import matplotlib.pyplot as plt
import pandas as pd
from gradio.themes.utils import fonts
import torch
import torchvision
from torchvision import transforms
from keras.preprocessing.image import ImageDataGenerator
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

class Load_Data():
    def __init__(self, dataurl ,drop_option, target):
        self.url = dataurl
        self.drop_option = drop_option
        self.target_col = target

    def data_to_csv(self):
        df = pd.read_csv(self.url)
        return df
    def load_image_data_keras(self, class_type):
        img_generator = ImageDataGenerator()
        img_data = img_generator.flow_from_directory(self.url, class_mode=class_type)
        return img_data
    def load_image_data_torch(self):
        trans = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        img_data = torchvision.datasets.ImageFolder(root=self.url, transforms = trans)
        return img_data
    def get_target(self, data):
        target_data = data[self.target_col]
        target_data = pd.DataFrame(target_data)
        return target_data
    def anal_data(self, data):
        describe_data = data.describe(include='all')
        numeric = data.select_dtypes(include='number').shape[1]
        category = data.select_dtypes(include='object').shape[1]
        nan_count = data.isnull().sum().sum()
        out_count = 0
        for column in data.columns:
            q3 = data[column].quantile(0.75)
            q1 = data[column].quantile(0.25)
            iqr = q3-q1
            outlier = data[column] > q3+3*iqr or data[column] < q1-3*iqr
            out_count = out_count+len(outlier)
        return describe_data, numeric, category, nan_count, out_count



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

    def process_nan(self, data, process_type, fill_type):
        nan_count = data.isnull().sum().sum()
        data_num = len(data)
        nan_ratio = nan_count/data_num*100
        if nan_ratio < 10 :
            data = data.dropna()
        elif nan_ratio>10 and process_type == "삭제":
            print("nan_count are many replace other way")
            data = data.fillna(method='ffill')
        elif process_type == "채우기":
            if fill_type == "평균":
                data = data.fillna(data.mean())
            elif fill_type == "앞의 값 대체":
                data = data.fillna(method='ffill')
            elif fill_type == "뒤의 값 대체":
                data = data.fillna(method='bfill')
            elif fill_type == "비례하는 값 대체":
                data = data.interpolate()
            elif fill_type == "시간 index 기준으로 대체":
                data = data.interpolate(method = 'time')
        return data

    def process_outlier(self, data, process_type, z = 3):
        del_outlier = copy.deepcopy(data)
        z_value = z
        if process_type == "IQR":
            for column in del_outlier.columns:
                q3 = del_outlier[column].quantile(0.75)
                q1 = del_outlier[column].quantile(0.25)
                iqr = q3 - q1
                outlier = del_outlier[column] > q3 + 3 * iqr or del_outlier[column] < q1 - 3 * iqr
                a = del_outlier[outlier].index
                del_outlier = del_outlier.drop(a, inplace = True)
        elif process_type == "z-score":
            for column in del_outlier.columns:
                outlier = del_outlier[abs(del_outlier[column]-np.mean(del_outlier[column]))/np.std(del_outlier[column])>z_value].index
                del_outlier = del_outlier.drop(outlier, inplace = True)
        return del_outlier




if __name__ == "__main__":
    target = "Radiation"
    data_load_anlysis = Load_Data("C:\\Users\\sin\\PycharmProjects\\sibal\\data\\SolarPrediction.csv", "삭제", target)
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