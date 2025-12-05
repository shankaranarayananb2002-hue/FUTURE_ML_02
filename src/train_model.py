
import argparse, os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def main(args):
    df = pd.read_csv(args.data)
    label_col = 'Churn Value'

    if label_col not in df.columns:
        raise ValueError(f'Dataset must include a {label_col} column as label')

    X = df.drop(label_col, axis=1)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model, args.model_path)

    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"ROC AUC: {auc}\n\n")
        f.write(report + '\n')
        f.write('Confusion Matrix:\n' + str(cm) + '\n')


    out_df = pd.DataFrame({'y_true': y_test, 'y_pred': preds, 'y_prob': probs})
    out_df.to_csv(os.path.join(args.results, 'predictions_sample.csv'), index=False)

    print('Training complete. Metrics saved to', args.results)
    print('Model saved to', args.model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to preprocessed CSV')
    parser.add_argument('--model_path', required=True, help='Path to save model .pkl')
    parser.add_argument('--results', required=True, help='Directory to save results')
    args = parser.parse_args()
    main(args)
