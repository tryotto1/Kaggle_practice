# library import
from torch.autograd import Variable
import sys

# module import
sys.path.append('/home/sykim/Desktop/project_refactorize/src/')
from features.mixup_data import mixup_data, mixup_criterion

def train_one_epoch(model, criterion, train_loader, optimizer, mixup_loss, accumulation_step=2): 
    model.train()
    train_loss = 0.
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(train_loader):
            
        inputs, targets = inputs.cuda(), targets.cuda()

        if mixup_loss:
            use_cuda=True
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.5, use_cuda = use_cuda) # alpha in [0.4, 1.0] 선택 가능
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs.cuda(), targets_a.cuda(), targets_b.cuda(), lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        
        if accumulation_step:
            if (i+1) % accumulation_step == 0:  
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
        

        train_loss += loss.item() / len(train_loader)
        
    return train_loss