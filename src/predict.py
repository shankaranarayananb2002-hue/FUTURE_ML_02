
import argparse
import pandas as pd
import joblib
import os

def main(args):
    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    label_col = 'Churn Value'

    if label_col in df.columns:
        X = df.drop(label_col, axis=1)
    else:
        X = df.copy()

    probs = model.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    out = X.copy()
    out['churn_probability'] = probs
    out['churn_pred'] = preds
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    print('Predictions saved to', args.output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to trained model .pkl')
    parser.add_argument('--input', required=True, help='Path to input CSV for predictions')
    parser.add_argument('--output', required=True, help='Path to save predictions CSV')
    args = parser.parse_args()
    main(args)
