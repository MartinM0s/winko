from sklearn.metrics import ConfusionMatrixDisplay
def predict_ds(X_train, X_test, transformer):
  _x_train_transform = transformer.transform(X_train)
  _x_test_transform = transformer.transform(X_test)
  return _x_train_transform, _x_test_transform
def prin_classification_report(y_train, y_train_pred,
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