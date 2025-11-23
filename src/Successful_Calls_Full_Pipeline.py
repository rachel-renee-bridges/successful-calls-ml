# ============================================================
# SUCCESSFUL CALLS — FULL PIPELINE SRC
# ============================================================

# ------------------------------
# 1. Imports
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# ------------------------------
# 2. Load Dataset and One-Hot Encode
# ------------------------------
df = pd.read_csv("INSERT FILE.csv")   # Replace with your CSV

# One-hot encode all categorical features
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop('INSERT TARGET', axis=1)   # Replace with your target
y = df['INSERT TARGET']


# ------------------------------
# 3. Train/Test Split (stratified)
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ------------------------------
# 4. Downsampling
# ------------------------------
print("Class counts in y_train before balancing:")
print(y_train.value_counts(), "\n")

if y_train.value_counts().min() == 0:
    print("WARNING: Only one class is present in y_train. Downsampling skipped.")
    X_train_balanced = X_train.copy()
    y_train_balanced = y_train.copy()
else:
    minority_class = y_train.value_counts().idxmin()
    majority_class = y_train.value_counts().idxmax()

    minority = X_train[y_train == minority_class]
    majority = X_train[y_train == majority_class]

    n_minority = len(minority)
    print(f"Minority class count: {n_minority}")
    print(f"Majority class will be downsampled to: {n_minority}\n")

    majority_downsampled = majority.sample(n=n_minority, random_state=42)

    X_train_balanced = pd.concat([majority_downsampled, minority], axis=0)
    y_train_balanced = pd.concat([
        pd.Series([majority_class] * n_minority),
        pd.Series([minority_class] * n_minority)
    ], axis=0)

    X_train_balanced.reset_index(drop=True, inplace=True)
    y_train_balanced.reset_index(drop=True, inplace=True)

print("Balanced class counts:")
print(y_train_balanced.value_counts(), "\n")


# ------------------------------
# 5. Scaling
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)


# ------------------------------
# 6. Define Models & Parameter Grids
# ------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt_grid_params = {"max_depth": [3,5,8,None], "min_samples_split":[2,5,10]}

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf_grid_params = {"n_estimators":[100,300], "max_depth":[None,5,10], "min_samples_split":[2,5]}

# Logistic Regression
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr_grid_params = {"C":[0.01,0.1,1,10,100], "penalty":["l2"], "solver":["lbfgs"]}

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                   solver="adam", max_iter=500, random_state=42)
nn_grid_params = {"alpha":[0.0001,0.001,0.01], "learning_rate_init":[0.001,0.01]}


# ------------------------------
# 7. Run GridSearch
# ------------------------------
def run_grid(model, params, X, y):
    gs = GridSearchCV(model, params, cv=kf, scoring="f1", n_jobs=-1)
    gs.fit(X, y)
    return gs

grids = {}
grids["Decision Tree"] = run_grid(dt, dt_grid_params, X_train_balanced, y_train_balanced)
grids["Random Forest"] = run_grid(rf, rf_grid_params, X_train_balanced, y_train_balanced)
grids["Logistic Regression"] = run_grid(lr, lr_grid_params, X_train_scaled, y_train_balanced)
grids["Neural Network"] = run_grid(nn, nn_grid_params, X_train_scaled, y_train_balanced)



# ------------------------------
# 8. Evaluation (F1 only)
# ------------------------------
def evaluate_model(name, grid_obj, X_test, y_test):
    print("\n====================================")
    print("Model:", name)
    print("Best Params:", grid_obj.best_params_)
    print("Best CV F1:", round(grid_obj.best_score_, 3))

for name, grid in grids.items():
    # Use scaled data for LR and NN
    if name in ["Logistic Regression", "Neural Network"]:
        evaluate_model(name, grid, X_test_scaled, y_test)
    else:
        evaluate_model(name, grid, X_test, y_test)



# ------------------------------
# 9. VISUALS
# ------------------------------
# Visual 1: Subscription Status
plt.figure(figsize=(6,4))
df['INSERT TARGET'].value_counts().plot(kind='bar')   # Replace with your target
plt.title("Subscription Status Distribution")
plt.xlabel("Subscription Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Visual 2: Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (After Feature Engineering)")
plt.show()

# Visual 3: Feature Importances (Random Forest) - Top 10 Features   #  Replace with your best model
rf_best = grids["Random Forest"].best_estimator_
importances = pd.Series(rf_best.feature_importances_, index=X.columns)

# Sort and take top 10
top_importances = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_importances.plot(kind='bar', color='steelblue')
plt.title("Top 10 Important Features (Random Forest)")
plt.ylabel("Importance Score")
plt.xlabel("Feature Name")
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()
