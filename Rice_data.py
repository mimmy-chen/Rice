import splitfolders
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

Data_path = "Rice"

splitfolders.ratio(Data_path, output='Rice_Train_Validation_Test', seed=1, ratio=(0.7,0.2,0.1))

batch_size = 10
image_size = (250, 250)
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


train_dataset = ImageFolder('Rice_Train_validation_Test/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataset = ImageFolder('Rice_Train_validation_Test/val', transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)