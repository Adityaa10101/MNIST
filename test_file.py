import torchvision
import torchvision.transforms as transforms
import json

testing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

test_set = torchvision.datasets.MNIST(root='./root', train=False, download=True, transform=testing)

image, label = test_set[0]  # grab the first test image
pixel_values = image.view(-1).tolist()  # flatten to 784 floats

print(f"Actual label: {label}")
print(json.dumps({"pixel_values": pixel_values}))