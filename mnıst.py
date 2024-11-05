import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time


device = torch.device("cuda")
print(f"Device: {device}")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_and_save_model(model, device, train_loader, criterion, optimizer, epochs=10, save_path="model.pth"):
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            
            optimizer.zero_grad()
            
           
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    end_time = time.time()
    print(f"Training completed in: {end_time - start_time:.2f} seconds")
    
    # Modeli kaydet
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Eğitim işlemini başlat ve modeli kaydet
train_and_save_model(model, device, train_loader, criterion, optimizer, epochs=10, save_path="simple_net_mnist.pth")
