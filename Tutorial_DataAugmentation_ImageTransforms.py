# Imports
import torch
import torchvision.transforms as transforms # Transformations we can perform on our datset
from torchvision.utils import save_image
from Tutorial_customDataset_image import CatsAndDogsDataset

# load Data
#my_transforms = transforms.ToTensor()

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0,1.0,1.0]) # (value - mean) / std ; Note ; this code does nothing!

])

dataset = CatsAndDogsDataset(csv_file= 'dataset/cats_dogs.csv', root_dir = 'dataset/cats_dogs_resized' , transform = my_transforms)

# train_loader = DataLoader()


img_num = 0
for _ in range(10):
    for img, label in dataset:
        save_image(img, 'img'+str(img_num)+'.png')
        img_num += 1


