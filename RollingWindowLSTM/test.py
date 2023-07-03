import torch

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            test_samples += inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    avg_test_loss = test_loss / test_samples
    accuracy = correct / test_samples
    
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
