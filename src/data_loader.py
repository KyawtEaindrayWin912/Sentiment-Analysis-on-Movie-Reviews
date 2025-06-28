import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="../data/processed_data.csv"):
    """
    Loads the preprocessed data from a CSV file.

    Parameters:
        path (str): Path to the processed CSV file.

    Returns:
        X (Series): Cleaned review texts
        y (ndarray): Sentiment labels (0 = negative, 1 = positive)
    """
    df = pd.read_csv(path)
    X = df['clean_review']
    y = df['sentiment_label'].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and testing sets.

    Parameters:
        X (Series): Input texts
        y (ndarray): Target labels
        test_size (float): Proportion of the data to use as test set
        random_state (int): Random seed

    Returns:
        X_train, X_test, y_train, y_test (tuples): Split datasets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
