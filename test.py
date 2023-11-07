import gradio as gr
import pandas as pd

# 예제 DataFrame 생성
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 22, 35, 28],
    "City": ["New York", "Los Angeles", "Chicago", "Houston", "Miami"]
}
df = pd.DataFrame(data)

def display_info(df):
    return df.info()

def display_description(df):
    return df.describe()

iface = gr.Interface(
    [gr.Data(name="data", type="dataframe")],
    [
        gr.Output(name="info", type="text"),
        gr.Output(name="description", type="text")
    ],
    title="Pandas DataFrame Info and Describe",
    description="Display Pandas DataFrame info and describe results.",
    examples=[{"data": df}]
)

iface.launch()