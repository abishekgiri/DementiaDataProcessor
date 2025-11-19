from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import shap
import joblib

# ==============================
# 1. DATA PROCESSOR
# ==============================

class DementiaDataProcessorCore:
    def __init__(self, demographics_path, predictions_path):
        self.demographics_path = Path(demographics_path)
        self.predictions_path = Path(predictions_path)
        self.data = None
        self.features = None
        self.target = None
        self.feature_cols = ["Age_x", "M/F_x", "EDUC", "SES_x", "MMSE_x", "CDR_x"]

    def load_and_preprocess(self):
        # Load Excel files
        data1 = pd.read_excel(self.demographics_path)
        data2 = pd.read_excel(self.predictions_path)

        # Merge on Subject ID
        self.data = pd.merge(data1, data2, on="Subject ID", how="inner")

        # Drop missing & duplicate rows
        self.data = self.data.dropna().drop_duplicates()

        # Encode gender: M -> 1, F -> 0
        if self.data["M/F_x"].dtype == "object":
            self.data["M/F_x"] = self.data["M/F_x"].map({"M": 1, "F": 0})

        # Binary target: Demented = 1, others = 0
        self.data["target"] = self.data["Group_x"].apply(
            lambda x: 1 if str(x).strip().lower() == "demented" else 0
        )

        # Standardize numeric columns (except gender)
        numeric_cols = ["Age_x", "EDUC", "SES_x", "MMSE_x", "CDR_x"]
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

        self.features = self.data[self.feature_cols]
        self.target = self.data["target"].astype(int)

        return self.balance_dataset()

    def balance_dataset(self):
        # Downsample majority class to match minority
        demented = self.data[self.data["target"] == 1]
        non_demented = self.data[self.data["target"] == 0]

        non_demented_downsampled = resample(
            non_demented,
            replace=False,
            n_samples=len(demented),
            random_state=42,
        )

        balanced_data = pd.concat([demented, non_demented_downsampled])

        X = balanced_data[self.feature_cols].values
        y = balanced_data["target"].astype(int).values

        return X, y, balanced_data


# ==============================
# 2. CUSTOM DECISION TREE (FROM SCRATCH)
# ==============================

class Node:
    def __init__(self, gini, samples, value, left=None, right=None, feature=None, threshold=None):
        self.gini = gini
        self.samples = samples
        self.value = value
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold


class CustomDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None
        self.feature_importance = None

    @staticmethod
    def gini_impurity(y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)

    @staticmethod
    def split_data(X, y, feature, threshold):
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if n_samples <= 1 or depth == self.max_depth or len(np.unique(y)) == 1:
            leaf_value = int(np.argmax(np.bincount(y)))
            return Node(gini=self.gini_impurity(y), samples=len(y), value=leaf_value)

        best_gini = 1.0
        best_split = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini = (len(y_left) / n_samples) * self.gini_impurity(y_left) + \
                       (len(y_right) / n_samples) * self.gini_impurity(y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "X_left": X_left,
                        "y_left": y_left,
                        "X_right": X_right,
                        "y_right": y_right,
                    }

        if best_split is None:
            leaf_value = int(np.argmax(np.bincount(y)))
            return Node(gini=self.gini_impurity(y), samples=len(y), value=leaf_value)

        left = self.build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
        right = self.build_tree(best_split["X_right"], best_split["y_right"], depth + 1)

        return Node(
            gini=best_gini,
            samples=n_samples,
            value=None,
            left=left,
            right=right,
            feature=best_split["feature"],
            threshold=best_split["threshold"],
        )

    def fit(self, X, y):
        self.root = self.build_tree(X, y)
        self.calculate_feature_importance()

    def _predict_single(self, node, x_row):
        if node.value is not None:
            return node.value
        if x_row[node.feature] < node.threshold:
            return self._predict_single(node.left, x_row)
        else:
            return self._predict_single(node.right, x_row)

    def predict(self, X):
        return np.array([self._predict_single(self.root, row) for row in X])

    def calculate_feature_importance(self):
        n_features = 6
        self.feature_importance = np.zeros(n_features)

        def traverse(node, parent_gini=0, weight=1.0):
            if node is None or node.feature is None:
                return
            reduction = weight * max(parent_gini - node.gini, 0)
            self.feature_importance[node.feature] += reduction

            if node.left is not None:
                traverse(node.left, node.gini, weight * node.left.samples / node.samples)
            if node.right is not None:
                traverse(node.right, node.gini, weight * node.right.samples / node.samples)

        if self.root is not None:
            traverse(self.root, parent_gini=self.root.gini)

        total = np.sum(self.feature_importance)
        if total > 0:
            self.feature_importance /= total


# ==============================
# 3. EVALUATION UTILITIES
# ==============================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_confusion(cm, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Demented", "Demented"],
        yticklabels=["Non-Demented", "Demented"],
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(values, feature_names, title="Feature Importance"):
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_names = np.array(feature_names)[order]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(sorted_vals)), sorted_vals)
    plt.yticks(range(len(sorted_vals)), sorted_names)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==============================
# 4. EXPLORATORY DATA ANALYSIS
# ==============================

def run_eda(df, feature_cols, target_col="target"):
    print("\n=== BASIC INFO ===")
    print(df[feature_cols + [target_col]].describe(include="all"))

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[feature_cols + [target_col]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Distribution plots
    for col in feature_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue=target_col, kde=True, stat="density", common_norm=False)
        plt.title(f"Distribution of {col} by Dementia Status")
        plt.tight_layout()
        plt.show()


# ==============================
# 5. MODEL TRAINING & COMPARISON
# ==============================

def train_and_compare_models(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }

    metrics_table = []
    roc_curves = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        m = compute_metrics(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, auc)

        metrics_table.append(
            {
                "Model": name,
                "Accuracy": m["accuracy"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "AUC": auc,
            }
        )

    metrics_df = pd.DataFrame(metrics_table).sort_values("F1", ascending=False)

    print("\n=== MODEL COMPARISON (TEST SET) ===")
    print(metrics_df.to_string(index=False))

    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.show()

    best_row = metrics_df.iloc[0]
    best_name = best_row["Model"]
    best_model = models[best_name]

    print(f"\nBest model by F1: {best_name}")
    print("\nClassification report for best model:")
    print(classification_report(y_test, best_model.predict(X_test)))

    if "Random Forest" in models:
        rf = models["Random Forest"]
        importances = rf.feature_importances_
        plot_feature_importance(importances, feature_names, title="Random Forest Feature Importance")

    return best_name, best_model, metrics_df


# ==============================
# 6. CROSS-VALIDATION
# ==============================

def run_cross_validation(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
    print("\n=== Stratified 5-Fold Cross-Validation (Random Forest, F1) ===")
    print(f"Scores: {scores}")
    print(f"Mean F1: {scores.mean():.3f} ± {scores.std():.3f}")


# ==============================
# 7. PCA & CLUSTERING
# ==============================

def run_pca_and_clustering(X, y):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", alpha=0.8, edgecolor="k"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (Colored by Dementia Status)")
    plt.legend(*scatter.legend_elements(), title="Demented")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    scatter2 = plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", alpha=0.8, edgecolor="k"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA (Colored by K-Means Clusters)")
    plt.legend(*scatter2.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.show()


# ==============================
# 8. SHAP EXPLAINABILITY
# ==============================

def run_shap(best_model, X_train, feature_names):
    print("\n=== Running SHAP explainability on best model ===")
    shap.summary_plot = shap.summary_plot  # just to keep linters happy

    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_train)
    except Exception:
        explainer = shap.KernelExplainer(best_model.predict_proba, X_train[:50])
        shap_values = explainer.shap_values(X_train[:200])

    plt.title("SHAP Summary (Bar)")
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_train, feature_names=feature_names)


# ==============================
# 9. MAIN PIPELINE
# ==============================

def main():
    base_dir = Path(__file__).resolve().parent

    demographics_file = base_dir / "oasis_longitudinal_demographics.xlsx"
    predictions_file = base_dir / "Predictions.xlsx"

    processor = DementiaDataProcessorCore(
        demographics_path=demographics_file,
        predictions_path=predictions_file,
    )

    X, y, df_balanced = processor.load_and_preprocess()
    feature_names = ["Age", "Gender", "Education", "SES", "MMSE", "CDR"]

    # --- EDA ---
    run_eda(df_balanced, processor.feature_cols, target_col="target")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Custom Decision Tree ---
    print("\n=== Custom Decision Tree (from scratch) ===")
    custom_tree = CustomDecisionTree(max_depth=3)
    custom_tree.fit(X_train, y_train)
    y_pred_tree = custom_tree.predict(X_test)
    metrics_tree = compute_metrics(y_test, y_pred_tree)
    print("Accuracy:", metrics_tree["accuracy"])
    print("Precision:", metrics_tree["precision"])
    print("Recall:", metrics_tree["recall"])
    print("F1:", metrics_tree["f1"])

    plot_confusion(metrics_tree["confusion_matrix"], title="Custom Decision Tree Confusion Matrix")
    plot_feature_importance(
        custom_tree.feature_importance,
        feature_names,
        title="Custom Decision Tree Feature Importance",
    )

    # --- Sklearn models & comparison ---
    best_name, best_model, metrics_df = train_and_compare_models(
        X_train, X_test, y_train, y_test, feature_names
    )

    # --- Cross-validation ---
    run_cross_validation(X, y)

    # --- PCA + KMeans ---
    run_pca_and_clustering(X, y)

    # --- SHAP explainability for best model ---
    run_shap(best_model, X_train, feature_names)

    # --- Save best model ---
    model_path = base_dir / "best_dementia_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nBest model ({best_name}) saved to: {model_path}")


if __name__ == "__main__":
    main()
