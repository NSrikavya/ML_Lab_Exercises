from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Assume these are your predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Confusion Matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

# Display confusion matrices
def plot_cm(cm, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

plot_cm(cm_train, "Training Confusion Matrix")
plot_cm(cm_test, "Test Confusion Matrix")

# Classification Reports
print("Training Classification Report:")
print(classification_report(y_train, y_train_pred))

print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))
