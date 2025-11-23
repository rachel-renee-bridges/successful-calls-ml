Successful Calls Project

Objective:
Predict whether a customer will subscribe to a term deposit after a marketing call.


Project includes:
A. Data preprocessing:

- One-hot encoding of categorical variables

- Feature scaling

- Safe downsampling of imbalanced classes

B. 4 machine learning models:

- Random Forest

- Decision Tree

- Logistic Regression

- Neural Network (MLP)

C. Evaluation:

- F1 score only

D. Visualizations for interpretability:

- Class distribution

- Correlation heatmap (clean, annotated)

- Top 10 feature importance (Random Forest, vertical bar chart)



Folder structure:
- notebooks/ : Jupyter notebooks (for exploration, experiments, or demos)

- README.txt : this file

- references/ : reference materials

- requirements.txt : Python dependencies

- src/ : Python source file (complete pipeline: data loading, preprocessing, modeling, visualization, with placeholders for your own dataset)

- visualizations/ : plots generated using the original dataset


Note:
Original bank marketing data cannot be shared due to privacy. The pipeline is structured to show workflow and methodology. There are places in the pipeline to insert your own dataset if available.

Run Instructions:

1. Install requirements: pip install -r requirements.txt

2. Run main: python src/Successful_Calls_Full_Pipeline.py
