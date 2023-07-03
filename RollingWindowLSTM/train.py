import torch

def train(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    best_val_loss = float('inf')
    best_model_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)
        
        avg_train_loss = train_loss / train_samples
        
        print(f"Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)
                
            avg_val_loss = val_loss / val_samples
            
            print(f"Epoch {epoch+1}/{max_epochs}: Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state_dict = model.state_dict()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print("Early stopping! No improvement in validation loss for {} epochs.".format(patience))
                    break
    
    return best_model_state_dict
