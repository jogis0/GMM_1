import torch
from torchvision import models, transforms, datasets
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = datasets.ImageFolder(root='./data', transform=transformations)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.vgg16(weights='DEFAULT').to(device)
model.eval()

imagenet_classes = open("imagenet_classes.txt").read().splitlines()

label_map = {
    'car': [436, 468, 511, 661, 609, 627, 656, 675, 717, 734, 751, 817, 864, 555, 581],
    'airplane': [403, 404, 895, 908],
    'motorcycle': [665, 670, 671]
}

thresholds = {
    'car': 0.5,
    'airplane': 0.5,
    'motorcycle': 0.5
}

ground_truths = []
predictions = []

with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        for idx, probs in enumerate(probabilities):
            predicted_label = "Unknown"
            max_prob = 0
            true_label = dataset.classes[labels[idx]]
            # print(f"--Image {idx}: True label - {true_label}--")
            for category, indices in label_map.items():
                category_prob = sum(probs[i] for i in indices)

                if category_prob > thresholds[category] and category_prob > max_prob:
                    predicted_label = category
                    max_prob = category_prob
                # print(f"Category: {category}, Probability: {category_prob}")

            ground_truths.append(true_label)
            predictions.append(predicted_label)
            # print(f"Predicted label: {predicted_label}")

ground_truths = np.array(ground_truths)
predictions = np.array(predictions)

lab = ['car', 'airplane', 'motorcycle']
accuracy = accuracy_score(ground_truths, predictions)
recall = recall_score(ground_truths, predictions, average=None, labels=lab)
precision = precision_score(ground_truths, predictions, average=None, labels=lab)
f1 = f1_score(ground_truths, predictions, average=None, labels=lab)

print(f"Accuracy: {accuracy}")
print(f"Recall - car: {recall[0]:.2f}, airplane {recall[1]:.2f}, motorcycle {recall[2]:.2f}")
print(f"Precision - car: {precision[0]:.2f}, airplane {precision[1]:.2f}, motorcycle {precision[2]:.2f}")
print(f"F1 score - car: {f1[0]:.2f}, airplane {f1[1]}, motorcycle {f1[2]:.2f}")
