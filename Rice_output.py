import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import Rice_data
import Rice_train

train_acc_history = Rice_train.train_acc_history
train_loss_history = Rice_train.train_loss_history
val_acc_history = Rice_train.val_acc_history
val_loss_history = Rice_train.val_loss_history
model1= Rice_train.model1
validation_loader = Rice_data.validation_loader

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

axs[0].plot(train_acc_history, color="red", marker="o")
axs[0].plot(val_acc_history, color="blue", marker="h")
axs[0].set_title('Accuracy Comparison between Train & Validation Set')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(['Train', 'Validation'], loc="lower right")

# Plot train and validation loss
axs[1].plot(train_loss_history, color="red", marker="o")
axs[1].plot(val_loss_history, color="blue", marker="h")
axs[1].set_title('Loss Comparison between Train & Validation Set')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(['Train', 'Validation'], loc="upper right")

plt.tight_layout()
plt.show()

class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

model1.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    step = 0
    for images, labels in validation_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model1(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        step += 1
        print("test step: {}".format(step))

conf_matrix = confusion_matrix(all_labels, all_predictions)
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()