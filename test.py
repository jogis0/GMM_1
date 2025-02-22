import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.vgg16(pretrained=True).to(device)
model.eval()

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# label_map = {
#     'Car': ['beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
#             'cab, hack, taxi, taxicab',
#             'convertible',
#             'Model T',
#             'jeep, landrover',
#             'limousine, limo',
#             'minivan',
#             'moving van',
#             'pickup, pickup truck',
#             'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
#             'racer, race car, racing car',
#             'sports car, sport car',],
#     'Airplane': ['airliner', 'warplane, military plane'],
#     'Motorcycle': ['moped', 'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader']
# }

label_map = {
    'Car': [436, 468, 511, 661, 609, 627, 656, 675, 717, 734, 751, 817, 864],
    'Airplane': [404, 895],
    'Motorcycle': [665, 670, 671]
}


def get_predicted_class(predictions, label_map):
    for label, classes in label_map.items():
        for cls in classes:
            if cls in predictions:
                return label
    return None


img = Image.open('data/car/images/000eba40a5b0dce6.jpg')
output = model(transformations(img).unsqueeze(0).to(device))
with open('imagenet_classes.txt', 'r') as fid:
  class_names = fid.readlines()
predicted_class = get_predicted_class(class_names[torch.argmax(output)], label_map)

print(f'Predicted class: {class_names[torch.argmax(output)]}')
