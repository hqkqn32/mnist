import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Cihazı seçin
device = torch.device("cuda")
print(f"Device: {device}")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet().to(device)
model.load_state_dict(torch.load("simple_net_mnist.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),          
    transforms.Resize((28, 28)),     
    transforms.ToTensor(),           
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(model, image_path):
    image = Image.open(image_path)  
    image = transform(image).unsqueeze(0).to(device)  
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

example_images = ["4.png", "0.png"]  

for img_path in example_images:
    prediction = predict_image(model, img_path)
    print(f"Görüntü: {img_path}, Tahmin Edilen Sınıf: {prediction}")
    
    img = Image.open(img_path)
    plt.imshow(img, cmap="gray")
    plt.title(f"Tahmin: {prediction}")
    plt.axis("off")
    plt.show()


