import gradio as gr
import mlflow.sklearn
import pandas as pd

# Load model from MLflow
model_uri = "runs:/<your_run_id>/model"  # Replace with actual run ID or use mlflow.search_runs()
model = mlflow.sklearn.load_model(model_uri)

def predict(sqft, bedrooms):
    df = pd.DataFrame([[sqft, bedrooms]], columns=["sqft", "bedrooms"])
    price = model.predict(df)[0]
    return f"Predicted Price: ${price:,.2f}"

demo = gr.Interface(fn=predict,
                    inputs=[gr.Number(label="Square Feet"), gr.Number(label="Bedrooms")],
                    outputs="text",
                    title="House Price Predictor")

demo.launch()
