import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.inspection import permutation_importance


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def make_pipe(model):
    # Missing values bharna + scaling + model, sab ek pipeline mein
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def compare_models(X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, model in get_models().items():
        s = cross_validate(make_pipe(model), X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "Model": name,
            "Accuracy": s["test_accuracy"].mean(),
            "Precision": s["test_precision"].mean(),
            "Recall": s["test_recall"].mean(),
            "F1": s["test_f1"].mean(),
            "ROC-AUC": s["test_roc_auc"].mean(),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def evaluate_and_save(results_df, X, X_train, X_test, y_train, y_test, tag, labels):
    best_name = results_df.iloc[0]["Model"]
    pipe = make_pipe(get_models()[best_name])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Best model:", best_name)
    print(classification_report(y_test, y_pred, target_names=labels))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=labels)
    plt.title(f"{tag} - Confusion Matrix")
    plt.savefig(f"../assets/{tag}_confusion.png", bbox_inches="tight")
    plt.show()

    # Feature importance
    r = permutation_importance(pipe, X_test, y_test, n_repeats=20,
                               random_state=42, scoring="roc_auc")
    imp = pd.Series(r.importances_mean, index=X.columns).sort_values()
    imp.plot(kind="barh", figsize=(8, max(4, 0.3 * len(imp))))
    plt.title(f"{tag} - Feature Importance (Permutation)")
    plt.xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.savefig(f"../assets/{tag}_importance.png", bbox_inches="tight")
    plt.show()

    results_df.round(3).to_csv(f"../assets/{tag}_results.csv", index=False)
    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])

    # App ke liye har feature ki range
    meta = {}
    for f in X.columns:
        col = X[f].dropna()
        meta[f] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "is_int": bool((col % 1 == 0).all()),
        }

    joblib.dump({
        "pipeline": pipe,
        "features": list(X.columns),
        "meta": meta,
        "model_name": best_name,
        "test_auc": float(test_auc),
    }, f"../models/{tag}_model.joblib")
    print(f"Saved models/{tag}_model.joblib | Test ROC-AUC: {test_auc:.3f}")
    return pipe