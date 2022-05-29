from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.matrics import ClassificationReport

def print_classification_report(y_train, y_train_pred,
                               y_test, y_test_pred):
  print("Train Classifiacation Report:")
  print(classification_report(y_train, y_train_pred))
  print("\n")
  print("Test Classifiacation Report:")
  print(classification_report(y_test, y_test_pred))

def plot_conf_matrix(y_train, y_train_predict,
                      y_test, y_test_predict):
  ConfusionMatrixDisplay.from_predictions(y_train,
                                          y_train_predict,
                                          normalize='true')
  ConfusionMatrixDisplay.from_predictions(y_test,
                                          y_test_predict,
                                          normalize='true')