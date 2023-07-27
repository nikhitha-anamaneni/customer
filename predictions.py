import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def predict_with_random_forest(file_path, new_data=None):
    # Load the dataset
    dataset = pd.read_csv(file_path)
    dataset = dataset.drop(['payload', 'transaction'], axis=1)

    # Create a new column 'Group' based on 'vCore' and 'workers'
    dataset['Group'] = dataset['workers'].astype(str) + '-' + dataset['vCore'].astype(str)

    # Encode categorical variables
    le = preprocessing.LabelEncoder()
    dataset['vCore'] = le.fit_transform(dataset['vCore'])
    dataset['Group'] = le.fit_transform(dataset['Group'])

    # Split the dataset into independent variables (X) and dependent variable (y)
    X = dataset.drop('Group', axis=1)
    y = dataset['Group']

    # Preprocess the data to handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Retrieve the column names
    column_names = dataset.drop('Group', axis=1).columns.tolist()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the RandomForestClassifier model
    rfclassifier = RandomForestClassifier()
    rfclassifier.fit(X_train, y_train)

    # Predict for the test data
    y_pred = rfclassifier.predict(X_test)

    print("Predictions:")
    print(y_pred)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if new_data is not None:
        # Create a DataFrame for the new data
        new_df = pd.DataFrame([new_data], columns=column_names)

        # Set the default values for missing columns from the dataset
        for column in column_names:
            if column not in new_df.columns:
                default_value = dataset[column].mode()[0]
                new_df[column] = default_value

        # Preprocess the new data to handle missing values
        new_df = imputer.transform(new_df)

        # Predict using the trained model
        new_pred = rfclassifier.predict(new_df)

        # Inverse transform the predicted labels
        predicted_labels = le.inverse_transform(new_pred)
        print("Predicted Labels:")
        print(predicted_labels)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Random Forest Classifier Prediction')
parser.add_argument('file_path', type=str, help='Path to the dataset file')
# Add optional arguments for column values
dataset = pd.read_csv(parser.parse_known_args()[0].file_path)
for column in dataset.columns:
    if column not in ['payload', 'transaction', 'Group']:
        parser.add_argument(f'--{column}', type=float, help=f'Value for the {column} column', nargs='?')

args = parser.parse_args()

new_data = {}
for column in dataset.columns:
    if column not in ['payload', 'transaction', 'Group']:
        value = getattr(args, column)
        if value is not None:
            new_data[column] = value

predict_with_random_forest(args.file_path, new_data)
