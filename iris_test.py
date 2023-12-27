import unittest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from iris_classification import train_and_evaluate_model  # Replace 'your_script' with the actual module name

class TestRandomForestClassifier(unittest.TestCase):
    def test_model_accuracy(self):
        # Load the Iris dataset
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Random Forest classifier
        clf = RandomForestClassifier(random_state=42)

        # Train the model using the function from your script
        trained_model = train_and_evaluate_model(clf, X_train, y_train)

        # Make predictions on the test data
        y_pred = trained_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Assert that the accuracy is greater than or equal to 0.9 (90%)
        self.assertGreaterEqual(accuracy, 0.9)

if __name__ == '__main__':
    unittest.main()
