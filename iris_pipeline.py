import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_iris_pipeline():
    # 1. Load and explore dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    print(f"Dataset Shape: {X.shape}")
    print(f"Class Distribution: {np.bincount(y)}")
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Define pipelines for multiple classifiers
    pipelines = {
        'KNN': Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),
        'SVM': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=42))]),
        'RandomForest': Pipeline([('classifier', RandomForestClassifier(random_state=42))])
    }
    
    # 4. Model Training and Cross-Validation Evaluation
    model_performances = {}
    for name, pipeline in pipelines.items():
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        model_performances[name] = {
            'CV Mean Accuracy': cv_scores.mean(),
            'Test Accuracy': test_acc
        }
        print(f"\n--- {name} Performance ---")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Test Set Accuracy: {test_acc:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    # 5. Hyperparameter Tuning for Best Model (SVM Example)
    print("\n--- Tuning SVM Hyperparameters ---")
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__kernel': ['linear', 'rbf']
    }
    
    grid_search = GridSearchCV(pipelines['SVM'], param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    final_preds = best_model.predict(X_test)
    print(f"Tuned Model Test Accuracy: {accuracy_score(y_test, final_preds):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))

if __name__ == "__main__":
    run_iris_pipeline()