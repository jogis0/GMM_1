from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load your image
image = Image.open("data/airplane/images/00a21fb1ed2af5e6.jpg")  # Replace with your image path

# Define the normalization transformation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Create a transform pipeline including ToTensor and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Apply the transformation to the image
transformed_image = transform(image)

# Convert the transformed image back to a displayable format
# Note: Denormalization might not be perfectly accurate, but it gives an idea
# of the transformed image's appearance.
def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    _tensor = tensor.numpy()
    _tensor = (_tensor * std[:, None, None] + mean[:, None, None])
    _tensor = _tensor.clip(0, 1) # clip values to [0, 1]
    return _tensor

denormalized_image = denormalize(transformed_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Display the original and normalized images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Normalized Image (Approximation)")
plt.imshow(transformed_image.transpose(1, 2, 0)) # Transpose to (H, W, C)
plt.axis("off")

plt.show()