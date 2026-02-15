from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from pipeline.data_pipeline import load_data, split_features_target, built_pipeline

MODEL_PATH = Path("model/pipeline.joblib")

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

def main() -> None:
    df = load_data()
    x, y = split_features_target(df)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )
    
    pipeline = built_pipeline(numeric_features=list(x.columns))
    pipeline.fit(x_train, y_train)
    
    y_pred = pipeline.predict(x_test)
    metrics = evaluate_classification(y_test, y_pred)

    print("Metricas de evaluacion")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision:{metrics['precision']:.4f}")
    print(f"  - Recall:   {metrics['recall']:.4f}")
    print(f"  - F1:       {metrics['f1']:.4f}")
    print("  - Matriz de confusión:")
    print(metrics["confusion_matrix"])
    print(classification_report(y_test, y_pred, target_names=["No diabetes", "Diabetes"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
    
    