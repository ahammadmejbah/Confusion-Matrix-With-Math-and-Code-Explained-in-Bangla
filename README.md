<div align="center">
      <h1> <img src="https://github.com/BytesOfIntelligences/BytesOfIntelligences/blob/main/Bytes%20Of%20Intelligencesshh.png" width="400px"><br/>Confusion Matrix With Math and Code  Explained in Bangla  সম্পূর্ণ বাংলায়</h1>
     </div>
<p align="center"> <a href="https://github.com/BytesOfIntelligences/" target="_blank"><img alt="" src="https://img.shields.io/badge/Website-EA4C89?style=normal&logo=dribbble&logoColor=white" style="vertical-align:center" /></a> <a href="@Ahammadmejbah" target="_blank"><img alt="" src="https://img.shields.io/badge/Twitter-1DA1F2?style=normal&logo=twitter&logoColor=white" style="vertical-align:center" /></a> <a href="https://facebook.com/ahammedmejbah" target="_blank"><img alt="" src="https://img.shields.io/badge/Facebook-1877F2?style=normal&logo=facebook&logoColor=white" style="vertical-align:center" /></a> <a href="https://www.instagram.com/ahammadmejbah/" target="_blank"><img alt="" src="https://img.shields.io/badge/Instagram-E4405F?style=normal&logo=instagram&logoColor=white" style="vertical-align:center" /></a> <a href="https://www.linkedin.com/in/ahammadmejbah/}" target="_blank"><img alt="" src="https://img.shields.io/badge/LinkedIn-0077B5?style=normal&logo=linkedin&logoColor=white" style="vertical-align:center" /></a> </p>

# Description
A confusion matrix is a table that is used to describe the performance of a classification model. It's particularly useful for evaluating the performance of machine learning algorithms, such as binary or multiclass classifiers. The confusion matrix is typically a 2x2 table for binary classification, but it can be extended to accommodate more classes in multiclass problems. The matrix contains four values:

# Features
- True Positives (TP): The number of instances that were correctly predicted as positive (i.e., the model predicted "yes" when the actual class was "yes").
- True Negatives (TN): The number of instances that were correctly predicted as negative (i.e., the model predicted "no" when the actual class was "no").
- False Positives (FP): The number of instances that were incorrectly predicted as positive (i.e., the model predicted "yes" when the actual class was "no").
- False Negatives (FN): The number of instances that were incorrectly predicted as negative (i.e., the model predicted "no" when the actual class was "yes").


Here are the steps to generate and calculate a confusion matrix:

**Step 1: Gather Data**
Collect the actual class labels and the predicted class labels for a set of data points. You should have two lists: one for the actual labels and one for the predicted labels.

**Step 2: Create the Matrix**

|                 | Actual Positive (Yes) | Actual Negative (No) |
|-----------------|------------------------|------------------------|
| Predicted Positive (Yes) | True Positives (TP)  | False Positives (FP) |
| Predicted Negative (No)  | False Negatives (FN) | True Negatives (TN)  |

**Step 3: Calculate the Values**
- True Positives (TP): Count the number of instances where both the actual and predicted labels are positive.
- True Negatives (TN): Count the number of instances where both the actual and predicted labels are negative.
- False Positives (FP): Count the number of instances where the actual label is negative, but the predicted label is positive.
- False Negatives (FN): Count the number of instances where the actual label is positive, but the predicted label is negative.

**Step 4: Use the Confusion Matrix**
Once you have calculated these values, you can use them to calculate various evaluation metrics, such as accuracy, precision, recall, and F1-score.

- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- Recall (Sensitivity): TP / (TP + FN)
- Specificity: TN / (TN + FP)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)

These metrics provide insights into the model's performance in terms of correctly and incorrectly classified instances.

Keep in mind that in multiclass problems, the confusion matrix will have more rows and columns to account for multiple classes. The principles are the same, but there are more values to calculate and interpret.

``` python
def custom_confusionMatrix(actual_value, predicted_value):
    if len(actual_value) != len(predicted_value):
        raise ValueError("The number of actual Value and The predicted value must be same !")
    
    #Intial the all parameter value is = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for actual, predicted in zip(actual_value, predicted_value):
        if actual == 1:
            if predicted == 1:
                TP += 1 # TP = TP+1
            else:
                FN +=1
        else:
            if predicted == 1:
                FP += 1
            else:
                TN += 1

    return [[TP, FN], [FP, TN]]

actual_value = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
predicted_value  =[1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
confusion_matrix = custom_confusionMatrix(actual_value, predicted_value)



for rowdata in confusion_matrix:
    print(rowdata)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix, annot=True, fmt = "d", cmap = "Blues", xticklabels=["Positive", "Negative"],
            yticklabels=["Positive", "Negative"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```

``` python

def custom_confusionMatrix(actual_value, predicted_value):
    if len(actual_value) != len(predicted_value):
        raise ValueError("The number of actual Value and The predicted value must be same !")
    
    #Intial the all parameter value is = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for actual, predicted in zip(actual_value, predicted_value):
        if actual == 1:
            if predicted == 1:
                TP += 1 # TP = TP+1
            else:
                FN +=1
        else:
            if predicted == 1:
                FP += 1
            else:
                TN += 1

    return TP, FN, FP, TN

actual_value = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
predicted_value  =[1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
TP, FN, FP, TN = custom_confusionMatrix(actual_value, predicted_value)

print("\n-----------------------------")
print("True Positive Value: ", TP)
print("True Negative Value: ", TN)
print("False Positive Value: ", FP)
print("False Negative Value: ", FN)



recall = TP/(TP + FN)
precision = TP / (TP + FP)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
negative_predicted_value = TN / (TN + FN)
print("\n-----------------------------")
print("Recall Value: ", recall)
print("Precission Value: ", precision)
print("Specificity Value: ", specificity)
print("Accuracy Value: ", accuracy)
print("Negative Predicted Value: ", negative_predicted_value)

```

``` python

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
actual_value = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
predicted_value  =[1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
confusions_matrix = confusion_matrix(actual_value, predicted_value)

TP = confusions_matrix[1, 1]
TN = confusions_matrix[0, 0]
FP = confusions_matrix[0, 1]
FN = confusions_matrix[1, 0]

print("\n-----------------------------")
print("True Positive Value: ", TP)
print("True Negative Value: ", TN)
print("False Positive Value: ", FP)
print("False Negative Value: ", FN)


accuracy_scores = accuracy_score(actual_value, predicted_value)
precision_scores = precision_score(actual_value, predicted_value)
recall_scores = recall_score(actual_value, predicted_value)
f1_scores = f1_score(actual_value, predicted_value)

print("\n-----------------------------")
print("Recall Value: ", recall_scores)
print("Precission Value: ", precision_scores)
print("Accuracy Value: ", accuracy_scores)
print("F1 Score Value: ", f1_scores)

```

``` python

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
sns.heatmap(confusions_matrix, annot=True, fmt = "d", cmap = "Blues", xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

```
    
