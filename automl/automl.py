from pycaret.classification import setup as classification_setup
from pycaret.regression import setup as regression_setup
from pycaret.classification import *
from pycaret.regression import *
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML
import pandas as pd





url = "C:\\Users\\sin\\PycharmProjects\\sibal\\automlboard\\pycaret.csv"
def pycaret_w (task, target, split ,copy_data):
    train_data, test_data = train_test_split(copy_data, test_size=split, random_state=1)
    url = "C:\\Users\\sin\\PycharmProjects\\sibal\\automlboard\\pycaret.csv"
    if task == "Regressor":
        set = regression_setup(data=train_data, target=target, test_data=test_data, normalize=True, transformation=False)
        best_model = set.compare_models(exclude=['arima', 'auto_arima'],budget_time=0.5)
        result_df = pull()
        result_df.to_csv(url, mode="w", index=False)
        print("done")
    # elif task == "Classification":
    #     return pass

def h2o_w (task, target, split, copy_data):
    url = "C:\\Users\\sin\\PycharmProjects\\sibal\\automlboard\\h2o.csv"
    if task == "Regressor":
        h2o.init()
        feature = copy_data.drop(target, axis=1)
        label = copy_data[target]
        y = target
        x = list(copy_data.columns)
        x.remove(y)
        train, test = train_test_split(copy_data, test_size=split)
        h2o_train = h2o.H2OFrame(train)
        h2o_test = h2o.H2OFrame(test)
        # h2o_train[y] = h2o_train[y].asfactor()
        # h2o_test[y] = h2o_test[y].asfactor()
        aml = H2OAutoML(max_runtime_secs=60)
        aml.train(x=x, y=y, training_frame=h2o_train, leaderboard_frame=h2o_test)
        leaderboard = aml.leaderboard
        result_df = leaderboard.as_data_frame()
        result_df.to_csv(url, mode="w", index=False)
        print("done")

class AutoMl_library():
    def __init__(self, x_train, x_test, y_train, y_test, split, task, target):
        #self.data = copy_data
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.split = split
        self.task = task
        self.target = target

    def pycaret_w(self, data):
        print("run_pycaret")
        train_data, test_data = train_test_split(data, test_size=self.split, random_state=1)
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\pycaret.csv"
        if self.task == "Regressor":
            set = regression_setup(data=train_data, target=self.target, test_data=test_data, normalize=True,
                                   transformation=False)
            best_model = set.compare_models(exclude=['arima', 'auto_arima'], budget_time=0.5)
            result_df = pull()
            result_df.to_csv(url, mode="w", index=False)
            print("done")
            return best_model

    def h2o_w(self, data):
        print("run_h2o")
        url = "C:\\Users\\sinb1\\PycharmProjects\\gradio demo\\automlboard\\h2o.csv"
        if self.task == "Regressor":
            h2o.init()
            feature = data.drop(self.target, axis=1)
            label = data[self.target]
            y = self.target
            x = list(data.columns)
            x.remove(y)
            train, test = train_test_split(data, test_size=self.split)
            h2o_train = h2o.H2OFrame(train)
            h2o_test = h2o.H2OFrame(test)
            # h2o_train[y] = h2o_train[y].asfactor()
            # h2o_test[y] = h2o_test[y].asfactor()
            aml = H2OAutoML(max_runtime_secs=60)
            aml.train(x=x, y=y, training_frame=h2o_train, leaderboard_frame=h2o_test)
            leaderboard = aml.leaderboard
            result_df = leaderboard.as_data_frame()
            result_df.to_csv(url, mode="w", index=False)
            print("done")
            return aml.leader.model_id