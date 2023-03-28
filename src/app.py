import gradio as gr
import pandas as pd
import os
import pickle
from datetime import datetime


# PAGE CONFIG
# page_icon = "💐"

# Setup variables and constants
# datetime.now().strftime('%d-%m-%Y _ %Hh %Mm %Ss')
DIRPATH = os.path.dirname(os.path.realpath(__file__))
tmp_df_fp = os.path.join(DIRPATH, "assets", "tmp",
                         f"history_{datetime.now().strftime('%d-%m-%Y')}.csv")
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "ml_components.pkl")
init_df = pd.DataFrame(
    {"petal length (cm)": [], "petal width (cm)": [],
     "sepal length (cm)": [], "sepal width (cm)": [], }
)

# FUNCTIONS


def load_ml_components(fp):
    "Load the ml component to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object


def setup(fp):
    "Setup the required elements like files, models, global variables, etc"

    # history frame
    if not os.path.exists(fp):
        df_history = init_df.copy()
    else:
        df_history = pd.read_csv(fp)

    df_history.to_csv(fp, index=False)

    return df_history


def make_prediction(df_input):
    """Function that take a dataframe as input and make prediction
    """
    global df_history

    print(f"\n[Info] Input information as dataframe: \n{df_input.to_string()}")
    df_input.drop_duplicates(inplace=True, ignore_index=True)
    print(f"\n[Info] Input with deplicated rows: \n{df_input.to_string()}")

    prediction_output = end2end_pipeline.predict_proba(df_input)
    print(
        f"[Info] Prediction output (of type '{type(prediction_output)}') from passed input: {prediction_output} of shape {prediction_output.shape}")

    predicted_idx = prediction_output.argmax(axis=-1)
    print(f"[Info] Predicted indexes: {predicted_idx}")
    df_input['pred_label'] = predicted_idx
    print(
        f"\n[Info] pred_label: \n{df_input.to_string()}")
    predicted_labels = df_input['pred_label'].replace(idx_to_labels)
    df_input['pred_label'] = predicted_labels
    print(
        f"\n[Info] convert pred_label: \n{df_input.to_string()}")

    predicted_score = prediction_output.max(axis=-1)
    print(f"\n[Info] Prediction score: \n{predicted_score}")
    df_input['confidence_score'] = predicted_score
    print(
        f"\n[Info] output information as dataframe: \n{df_input.to_string()}")
    df_history = pd.concat([df_history, df_input], ignore_index=True).drop_duplicates(
        ignore_index=True, keep='last')
    return df_history


def download():
    return gr.File.update(label="History File",
                          visible=True,
                          value=tmp_df_fp)


def hide_download():
    return gr.File.update(label="History File",
                          visible=False)


# Setup execution
ml_components_dict = load_ml_components(fp=ml_core_fp)
labels = ml_components_dict['labels']
end2end_pipeline = ml_components_dict['pipeline']
print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")
print(f"\n[Info] Predictable labels: {labels}")
idx_to_labels = {i: l for (i, l) in enumerate(labels)}
print(f"\n[Info] Indexes to labels: {idx_to_labels}")

df_history = setup(tmp_df_fp)


# APP Interface
with gr.Blocks() as demo:
    gr.Markdown('''<img class="center" src="https://www.thespruce.com/thmb/GXt55Sf9RIzADYAG5zue1hXtlqc=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/iris-flowers-plant-profile-5120188-01-04a464ab8523426fab852b55d3bb04f0.jpg"  width="50%" height="50%"> 
    <style>
    .center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%;
    }
    </style>''')
    gr.Markdown('''# 💐 Iris Classification App
    This app shows a simple demo of a Gradio app for Iris flowers classification.
    ''')

    df = gr.Dataframe(
        headers=["petal length (cm)",
                 "petal width (cm)",
                 "sepal length (cm)",
                 "sepal width (cm)"],
        datatype=["number", "number", "number", "number", ],
        row_count=3,
        col_count=(4, "fixed"),
    )
    output = gr.Dataframe(df_history)

    btn_predict = gr.Button("Predict")
    btn_predict.click(fn=make_prediction, inputs=df, outputs=output)

    # output.change(fn=)

    file_obj = gr.File(label="History File",
                       visible=False
                       )

    btn_download = gr.Button("Download")
    btn_download.click(fn=download, inputs=[], outputs=file_obj)
    output.change(fn=hide_download, inputs=[], outputs=file_obj)

if __name__ == "__main__":
    demo.launch(debug=True)
