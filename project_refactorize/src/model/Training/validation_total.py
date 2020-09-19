# library import
import numpy as np
import torch

# model import
from sklearn.metrics import f1_score

# validation
def validation(model, criterion, valid_loader, y_true, **valid_kwargs):    
    # get from kwags
    valid_df = valid_kwargs['valid_df']
    valid_dataset = valid_kwargs['valid_dataset']
    num_classes = valid_kwargs['num_classes']
    batch_size = valid_kwargs['batch_size']         

    # model code
    model.eval()
    valid_preds = np.zeros((len(valid_dataset), num_classes))
    val_loss = 0.
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):

            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = model(inputs).detach()
            loss = criterion(outputs, targets)
            valid_preds[i * batch_size: (i+1) * batch_size] = outputs.cpu().numpy()
            
            val_loss += loss.item() / len(valid_loader)
            
        y_pred = np.argmax(valid_preds, axis=1)
        val_score = f1_score(y_true, y_pred, average='micro')  
        
    return val_loss, val_score

