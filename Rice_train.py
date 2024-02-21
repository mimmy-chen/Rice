import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

import Rice_data
import Rice_CNN_model

train_loader = Rice_data.train_loader
validation_loader = Rice_data.validation_loader
model1 = Rice_CNN_model.RiceCNN().cuda()



loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model1.parameters(), lr=0.001)

train_step = 0
epoch = 20


train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []
writer = SummaryWriter("./log_Rice")

for i in range(epoch):
    print("------trianing time: {} ------".format(i+1))

    model1.train()

    total_train_loss = 0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model1(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1

        total_train += labels.size(0)
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()


        print("training step: {}, Loss: {}".format(train_step, loss.item()))
        writer.add_scalar("train_loss", loss.item(), train_step)

    train_loss = total_train_loss / len(validation_loader)
    train_acc = correct_train / total_train

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    model1.eval()

    total_val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.cuda()
            labels = labels.cuda()
            outputs = model1(images)
            loss = loss_fn(outputs, labels)

            total_val_loss += loss.item()
            total_val += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()

    print("total validation loss: {}".format(total_val_loss))
    writer.add_scalar("validation_loss", total_val_loss, i)

    val_loss = total_val_loss / len(validation_loader)
    val_acc = correct_val / total_val

    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

writer.close()