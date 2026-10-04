import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path

# Build paths from the file location
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "wine.csv"

def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Please add it during Phase 4.")
        return

    print("Loading data...")
    # Wine datasets often use semicolons; adjust if your specific csv uses commas
    df = pd.read_csv(DATA_PATH, sep=';') 

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()