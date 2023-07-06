import torch
import time

def train(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    best_val_loss = float('inf')
    best_model_state_dict = None
    epochs_without_improvement = 0

    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

            # Print progress
            progress = (batch_idx + 1) / len(train_loader) * 100
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{max_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Progress: {progress:.2f}%, Elapsed Time: {elapsed_time:.2f}s", end='\r')

        avg_train_loss = train_loss / train_samples

        print(f"\nEpoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
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

    total_time = time.time() - start_time
    print("Training complete! Total time: {:.2f}s".format(total_time))

    return best_model_state_dict