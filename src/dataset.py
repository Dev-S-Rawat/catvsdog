import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

data_transforms = transforms.Compose([
    transforms.Resize((224,224)), # mobileNet format 224x224
    transforms.ToTensor(), # conversion of PIL image to pytorch tensor
    transforms.Normalize( # Standard imagesNet normalization for mean and std. deviation
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.24, 0.225]
    )
])


class CatDogDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform =transform

    def __len__(self): # tells pytorch how many total images are in this dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # pytorch asks for a item at specific index
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # open image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label