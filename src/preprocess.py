import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(df, exclude=[]):
    le = LabelEncoder()
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype == 'object':
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = le.fit_transform(df[col])
    return df


def basic_cleaning(df):
    
    df.columns = df.columns.str.strip()
    
    df = df.replace(" ", np.nan)
    return df

def encode_categoricals(df, exclude=[]):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col in exclude:
            continue
        try:
            df[col] = df[col].fillna('NA').astype(str)
            df[col] = le.fit_transform(df[col])
        except Exception as e:
            print(f"Skipping encoding for {col}: {e}")
    return df

def main(args):
    df = pd.read_csv(args.input)
    df = basic_cleaning(df)
    
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
   
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
   
    label_col = 'Churn Value' if 'Churn' in df.columns else None
    df = encode_categoricals(df, exclude=[label_col] if label_col else [])
    
    if label_col:
        df[label_col] = df[label_col].astype(str).str.lower().map({'yes':1,'no':0,'1':1,'0':0}).fillna(0).astype(int)
    
    df.to_csv(args.output, index=False)
    print(f"Saved cleaned data to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw CSV')
    parser.add_argument('--output', required=True, help='Path to save cleaned CSV')
    args = parser.parse_args()
    main(args)
