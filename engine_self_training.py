import math
import sys
from typing import Iterable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from timm.utils import accuracy

def train_one_epoch(model: torch.nn.Module, args, train_config,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler, 
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, ((images_weak, images_strong, mask), targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None: 
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay 
        model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)
        
        images_weak, images_strong = images_weak.to(device, non_blocking=True), images_strong.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True) 
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():             
            # pseudo-label with ema model
            probs_ema = F.softmax(model_ema.ema(images_weak),dim=-1)
            
            score, pseudo_targets = probs_ema.max(-1)
            conf_mask = score>train_config['conf_threshold']
          
            pseudo_label_acc = (pseudo_targets[conf_mask] == targets[conf_mask]).float().mean().item()           
            conf_ratio = conf_mask.float().sum()/conf_mask.size(0)
            metric_logger.update(conf_ratio=conf_ratio)
            metric_logger.update(pseudo_label_acc=pseudo_label_acc)    
            
            
        with amp_autocast():    
            if args.mask:
                logits, x_recon, loss_align, mask = model(images_strong, mask=mask)
            else:
                logits = model(images_strong)
                
            # self-training loss    
            loss_st = F.cross_entropy(logits[conf_mask], pseudo_targets[conf_mask])

            # fairness regularization
            probs = F.softmax(logits,dim=-1) 
            probs_all = utils.all_gather_with_grad(probs)
            probs_batch_avg = probs_all.mean(0) # average prediction probability across all gpus

            if args.nb_classes>=512: 
                # moving average          
                if step==0:
                    probs_avg = probs_batch_avg
                else:    
                    probs_avg = 0.5*(probs_avg.detach()+probs_batch_avg)
                loss_fair = -(torch.log(probs_avg)).mean() / 0.5
            else:
                # batch average
                probs_avg = probs_batch_avg
                loss_fair = -(torch.log(probs_avg)).mean()   
            
            if args.mask:
                # mask image modeling loss
                loss_mim = F.l1_loss(x_recon, images_strong, reduction='none')                
                loss_mim = (loss_mim * mask).sum() / mask.sum() / images_strong.size(1)   
                
                # global-local feature alignment loss
                loss_align = torch.mean(loss_align)

                loss = loss_st + loss_fair + loss_mim + train_config['w_align'] * loss_align 
            else:
                loss = loss_st + loss_fair
            
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:                   
            loss.backward(create_graph=False)       
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()  

        metric_logger.update(loss_st=loss_st.item())
        metric_logger.update(loss_fair=loss_fair.item())
        if args.mask:
            metric_logger.update(loss_mim=loss_mim.item())        
            metric_logger.update(loss_align=loss_align.item())
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:            
            log_writer.update(loss_st=loss_st.item(), head="train")
            log_writer.update(loss_fair=loss_fair.item(), head="train")
            if args.mask:
                log_writer.update(loss_mim=loss_mim.item(), head="train")
                log_writer.update(loss_align=loss_align.item(), head="train")
    
            log_writer.update(conf_ratio=conf_ratio, head="train")
            log_writer.update(pseudo_label_acc=pseudo_label_acc, head="train")          
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None, args=None):
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if model_ema is not None:
        model_ema.ema.eval()   
        
    if args.dataset in ['pets', 'caltech101']:
        all_outputs = []
        all_ema_outputs = []
        all_targets = []
        
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # compute output
        output = model(images)
            
        if args.dataset in ['pets', 'caltech101']:
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())   
        else:    
            acc = accuracy(output, target)[0]
            metric_logger.meters['acc1'].update(acc.item(), n=images.shape[0])
        
        if model_ema is not None:
            ema_output = model_ema.ema(images) 
            
            if args.dataset in ['pets', 'caltech101']:
                all_ema_outputs.append(ema_output.cpu())
            else:  
                ema_acc1 = accuracy(ema_output, target)[0]  
                metric_logger.meters['ema_acc1'].update(ema_acc1.item(), n=images.shape[0])

    if args.dataset in ['pets', 'caltech101']:
        mean_per_class = utils.mean_per_class(torch.cat(all_outputs), torch.cat(all_targets))
        metric_logger.meters['acc1'].update(mean_per_class) 
        if model_ema is not None:
            mean_per_class = utils.mean_per_class(torch.cat(all_ema_outputs), torch.cat(all_targets))
            metric_logger.meters['ema_acc1'].update(mean_per_class) 
            
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1))    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

