# library import
import sys
import time

# torch import
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam, SGD, Optimizer

# module import
sys.path.append('/home/shared/sykim/lab_kaggle_practice1/project_refactorize/src/model')
from optimizer.AdamW import AdamW
from optimizer.CosineAnnealingWithRestartsLR import CosineAnnealingWithRestartsLR
from Training.train_one_epoch import train_one_epoch
from Training.validation_total import validation

# training code
def train_model(num_epochs=60, accumulation_step=4, mixup_loss=False, cv_checkpoint=False, fine_tune=False,
                weight_file_name='weight_best.pt', y_true=None, **train_kwargs):

    # get from "train_kwargs"
    train_loader = train_kwargs['train_loader']
    valid_loader = train_kwargs['valid_loader']
    model = train_kwargs['model']
    criterion = train_kwargs['criterion']

    # get from "train_kwargs" for validation
    valid_df = train_kwargs['valid_df']
    valid_dataset = train_kwargs['valid_dataset']
    num_classes = train_kwargs['num_classes']
    batch_size = train_kwargs['batch_size']    
    
    # choose scheduler
    if fine_tune:
        lr = 0.00001
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.000025)   
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    else:    
        lr = 0.01
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.025)
        eta_min = 1e-6
        T_max = 5
        T_mult = 2
        restart_decay = 1.0
        scheduler = CosineAnnealingWithRestartsLR(optimizer,T_max=T_max, eta_min=eta_min, T_mult=T_mult, restart_decay=restart_decay)
        
        train_result = {}
    
    print(weight_file_name)
    train_result['weight_file_name'] = weight_file_name
    
    best_epoch = -1
    best_score = 0.
    lrs = []
    score = []
    
    for epoch in range(num_epochs):
        
        start_time = time.time()        

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss, accumulation_step) 

        # valid_kwargs
        valid_kwargs = dict(
            valid_df = valid_df,
            num_classes = num_classes,            
            batch_size = batch_size,            
            valid_dataset = valid_dataset,
        )
        val_loss, val_score = validation(model, criterion, valid_loader, y_true, **valid_kwargs)
        
        score.append(val_score)
    
        # model save (score or loss?)
        if cv_checkpoint:
            if val_score > best_score:
                best_score = val_score
                train_result['best_epoch'] = epoch + 1
                train_result['best_score'] = round(best_score, 5)
                torch.save(model.state_dict(), weight_file_name)
        else:
            if val_loss < best_loss:
                best_loss = val_loss
                train_result['best_epoch'] = epoch + 1
                train_result['best_loss'] = round(best_loss, 5)
                torch.save(model.state_dict(), weight_file_name)

        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.4f}  val_loss: {:.4f}  cv_score: {:.4f}  lr: {:.6f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))
        
        for param_group in optimizer.param_groups:
            lrs.append(param_group['lr'])
                # scheduler update
        if fine_tune:
            if cv_checkpoint:
                scheduler.step(val_score)
            else:
                scheduler.step(val_loss)
        else:
            scheduler.step()
     
    return train_result, lrs, score

