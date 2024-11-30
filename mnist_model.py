import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.conv1_1x1 = nn.Conv2d(8, 6, 1)  # 28x28x6
        
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)  # 14x14x16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2_1x1 = nn.Conv2d(16, 12, 1)  # 14x14x12
        
        self.fc1 = nn.Linear(12 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1_1x1(x)
        x = F.max_pool2d(x, 2)  # 14x14x6
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv2_1x1(x)
        x = F.max_pool2d(x, 2)  # 7x7x12
        
        x = x.view(-1, 12 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    model = LightMNIST().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                         T_max=len(train_loader),
                                                         eta_min=1e-4)
    
    param_count = count_parameters(model)
    print(f"Total trainable parameters: {param_count}")
    
    model.train()
    warmup_steps = 200
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        if batch_idx < warmup_steps:
            lr_scale = min(1., float(batch_idx + 1) / warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 0.003
                
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}')
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    torch.save(model.state_dict(), 'mnist_model.pth')
    return accuracy, param_count

if __name__ == "__main__":
    train_model() 