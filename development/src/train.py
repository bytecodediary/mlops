# training source code

import logging
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ])
logging.info("Training started...")
logger = logging.getLogger(__name__)

# import logging
# import mlflow
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("train.log"),
#         logging.StreamHandler()
#     ]
# )
# logging.info("Training started...")
# logger = logging.getLogger(__name__)

# logging.info("Loading iris dataset...")
# iris = load_iris()

# logging.info("Splitting dataset into train and test sets...")
# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, 
#                                                     test_size=0.2, random_state=42)

# with mlflow.start_run():
#     # Train model
#     logging.info("Training model...")
#     model = RandomForestClassifier()
#     model.fit(x_train, y_train)

#     # Evaluate model on test set
#     logging.info("Evaluating model on test set...")
#     predictions = model.predict(x_test)
#     accuracy = accuracy_score(y_test, predictions)
#     report = classification_report(y_test, predictions)
#     matrix = confusion_matrix(y_test, predictions)

#     # Log metrics for predictions
#     logging.info("Logging metrics for predictions...")
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.log_text("classification_report", report)
#     mlflow.log_text("confusion_matrix", str(matrix))
#     logging.info(f"Model accuracy: {accuracy}")
#     logging.info(f"Classification Report:\n{report}")
#     logging.info(f"Confusion Matrix:\n{matrix}")
logging.info("Loading iris dataset...")
iris = load_iris()

logging.info("Splitting dataset into train and test sets...")
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, 
                                                    test_size=0.2, random_state=42)

with mlflow.start_run():
    # Train model
    logging.info("Training model...")
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Evaluate model on test set
    logging.info("Evaluating model on test set...")
    predictions = model.predict(x_test)
    accurac = accuracy_score(y_test, predictions)

    # Log metrics for predictions
    logging.info("Logging metrics for predictions...")
    mlflow.log_metric("accuracy", accurac)
    logging.info(f"Model accuracy: {accurac}")
