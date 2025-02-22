import torch
from torchvision import models, transforms, datasets
import numpy as np

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

tp, fp, tn, fn = 0, 0, 0, 0
ground_truths = []
predictions = []

with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        model_predictions = torch.argmax(probabilities, dim=1)

        for idx, pred in enumerate(model_predictions.cpu().numpy()):
            predicted_label = None
            for category, indices in label_map.items():
                if pred in indices:
                    predicted_label = category
                    break
            if predicted_label is None:
                predicted_label = "Unknown"

            true_label = dataset.classes[labels[idx]]

            ground_truths.append(true_label)
            predictions.append(predicted_label)

            if predicted_label == true_label:
                tp += 1
            else:
                fp += 1

ground_truths = np.array(ground_truths)
predictions = np.array(predictions)

tn = np.sum(np.bitwise_and(predictions != ground_truths, ground_truths != "Unknown"))
fn = np.sum(np.bitwise_and(predictions == "Unknown", ground_truths != "Unknown"))

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (recall * precision) / (recall + precision)

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")
