## Neural Network Class
### Author: John Hawkins


This repository adapts and extends the core concepts learned in the deeplearning.ai Coursera "Neural Networks and Deep Learning" course in order to develop a custom vanilla feed forward neural network classifier Python class with a standard back propagation and gradient descent weight update. This new classifier is benchmarked against the TensorFlow Keras Sequential classifier model and the scikit-learn "MLPClassifier" model.

### Prerequisites

In order to recreate the results of this project Python 3.6 must be installed along with the following packages:

```
tensorflow==2.1.0
scikit-learn==0.21.2
numpy==1.16.4
h5py==2.9.0
pandas==0.24.2
```

### Running The Code

Follow the steps below to run the benchmark:

```
1. From terminal run "python run_benchmark.py"
```

### Red Wine Data Results

The following confusion matrices and classification precision and recall indicate that our custom neural network classes performs
within range of the TensorFlow Sequential classifier model and scikit-learn MLPClassifier model on the binary classification red wine quality dataset:

```
custom network confusion matrix:
[[114  35]
 [ 48 123]]
custom network classification report:
              precision    recall  f1-score   support

           0       0.70      0.77      0.73       149
           1       0.78      0.72      0.75       171

    accuracy                           0.74       320
   macro avg       0.74      0.74      0.74       320
weighted avg       0.74      0.74      0.74       320
custom network training time (seconds): 2.751539945602417


tensorflow network confusion matrix:
[[109  40]
 [ 41 130]]
tensorflow network classification report:
              precision    recall  f1-score   support

           0       0.73      0.73      0.73       149
           1       0.76      0.76      0.76       171

    accuracy                           0.75       320
   macro avg       0.75      0.75      0.75       320
weighted avg       0.75      0.75      0.75       320
tensorflow network training time (seconds): 101.92008113861084


scikit-learn network confusion matrix:
[[106  43]
 [ 34 137]]
scikit-learn network classification report:
              precision    recall  f1-score   support

           0       0.76      0.71      0.73       149
           1       0.76      0.80      0.78       171

    accuracy                           0.76       320
   macro avg       0.76      0.76      0.76       320
weighted avg       0.76      0.76      0.76       320
scikit-learn network training time (seconds): 12.519914865493774
```

### Cat Image Results

The following confusion matrices and classification precision and recall indicate that our custom neural network classes performs
within range of the TensorFlow Sequential classifier model and scikit-learn MLPClassifier model on the cat image binary classification data set:

```
custom network confusion matrix:
[[12  5]
 [ 8 25]]
custom network classification report:
              precision    recall  f1-score   support

           0       0.60      0.71      0.65        17
           1       0.83      0.76      0.79        33

    accuracy                           0.74        50
   macro avg       0.72      0.73      0.72        50
weighted avg       0.75      0.74      0.74        50
custom network training time (seconds): 35.964136362075806


tensorflow network confusion matrix:
[[12  5]
 [ 6 27]]
tensorflow network classification report:
              precision    recall  f1-score   support

           0       0.67      0.71      0.69        17
           1       0.84      0.82      0.83        33

    accuracy                           0.78        50
   macro avg       0.76      0.76      0.76        50
weighted avg       0.78      0.78      0.78        50
tensorflow network training time (seconds): 64.23010206222534


scikit-learn network confusion matrix:
[[12  5]
 [ 9 24]]
scikit-learn network classification report:
              precision    recall  f1-score   support

           0       0.57      0.71      0.63        17
           1       0.83      0.73      0.77        33

    accuracy                           0.72        50
   macro avg       0.70      0.72      0.70        50
weighted avg       0.74      0.72      0.73        50
scikit-learn network training time (seconds): 32.63913702964783
```
