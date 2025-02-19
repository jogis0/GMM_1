import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt


def read_img(file_name):
    img = io.imread(file_name)
    if img.ndim == 2:
      img = gray2rgb(img)
    img = [resize(img, (224, 224))]
    img = torch.tensor(img)
    img = img.permute(0, 3, 1, 2)
    return img.float()


classes = ["Airplane", "Car", "Motorcycle"]

transform = transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(20, interpolation=PIL.Image.BILINEAR),
    transforms.ToTensor()
])


class CustomDataset(Dataset):
    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.transforms = transforms

        # change number 1: change paths
        self.class1_files = glob.glob(self.images_dir + "/{}/images/*.jpg".format(classes[0].lower()))
        self.class2_files = glob.glob(self.images_dir + "/{}/images/*.jpg".format(classes[1].lower()))
        self.class3_files = glob.glob(self.images_dir + "/{}/images/*.jpg".format(classes[2].lower()))
        self.class1 = len(self.class1_files)
        self.class2 = len(self.class2_files)
        self.class3 = len(self.class3_files)

        self.files = self.class1_files + self.class2_files + self.class3_files

        # change number 2: if there was issues with number of samples set labels
        self.labels = np.zeros(len(self.files))
        self.labels[self.class1:] = 1

        # shuffle data
        self.order = [x for x in np.random.permutation(len(self.labels))]
        self.files = [self.files[x] for x in self.order]
        self.labels = [self.labels[x] for x in self.order]

    def __len__(self):
        return (len(self.labels))

    def __getitem__(self, i):
        files = self.files[i]

        im = read_img(files)[0]

        img = np.array(im.numpy())
        img = torch.tensor(img)

        y = self.labels[i]
        return (img, y)


# change number 1: change paths
dataset = CustomDataset("./data")

len(dataset)

for i in [0, 10]:
    sample = dataset[i]
    print(sample[0].shape, sample[1])

train_dataset = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

images, labels = next(iter(train_dataset))
images.shape

print(images[0].shape)
im = images[0].permute(2, 1, 0)
plt.imshow(im)
