import torch


def test_model(model, test_loader, criterion):
    # 1. Put the model in evaluation mode (disables dropout, fixes batchnorm, etc.)
    model.eval() 
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # 2. Tell PyTorch NOT to calculate gradients to save massive amounts of memory and time
    with torch.no_grad(): 
        # Optional: Wrap with tqdm for a progress bar here too!
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 3. Calculate Accuracy
            # outputs.max(1) returns the highest value and its index (which is our predicted class)
            _, predicted = outputs.max(1) 
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    # Calculate final metrics
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f"\n--- Test Results ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%\n")
    
    # Put the model back into training mode just in case you want to keep training!
    model.train() 
    
    return avg_loss, accuracy