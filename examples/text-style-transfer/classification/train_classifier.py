import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SIOP_Dataset, set_seed
from torch.utils.tensorboard import SummaryWriter
import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW


def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.to(device)

    
    set_seed(42)
    train_dataset = SIOP_Dataset('train')
    val_dataset = SIOP_Dataset('dev')

    # pre-training 
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_per_gpu*len(opt.gpu.split(',')), shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_per_gpu*len(opt.gpu.split(',')), shuffle=True, pin_memory=True)
    model.train()
    log_dir = "./runs/train_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    optimizer = AdamW(model.parameters(), lr=opt.lr, eps=1e-8)
    total_steps = len(train_loader) * opt.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=0, # Default value
        num_training_steps=total_steps)
    best_acc = None
    for i in range(opt.epochs):
        # training...
        train_losses = 0
        total = 0
        correct = 0
        for answer, label in tqdm(train_loader):
            optimizer.zero_grad()
            batch = tokenizer(answer, padding='max_length',  max_length=64, truncation=True, return_tensors="pt")
            batch['labels'] = label
            batch = batch.to(device)
            output = model(**batch)
            loss = output.loss
            correct += float((torch.argmax(output.logits, dim=-1)==batch['labels']).sum())
            total += len(answer)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses += loss
            # scheduler.step()
        epoch_loss = train_losses / len(train_loader)
        writer.add_scalar('Train/loss', epoch_loss, i)  
        writer.add_scalar('Train/acc', correct / total, i)  
        # writer.add_scalar('lr/lr', scheduler.get_last_lr()[0], i)  
        writer.add_scalar('lr/lr', opt.lr, i)  

        
        # validation
        val_losses = 0
        total = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for answer, label in tqdm(val_loader):
                optimizer.zero_grad()
                batch = tokenizer(answer, padding='max_length',  max_length=64, truncation=True, return_tensors="pt")
                batch['labels'] = label
                batch = batch.to(device)
                output = model(**batch)
                loss = output.loss
                correct += float((torch.argmax(output.logits, dim=-1)==batch['labels']).sum())
                total += len(answer)
                val_losses += loss
            epoch_loss = val_losses/ len(val_loader)
            epoch_acc = correct / total
            writer.add_scalar('Val/loss', epoch_loss, i) 
            writer.add_scalar('Val/acc', epoch_acc, i)  
        if i == 0 or epoch_acc > best_acc:  
            best_acc = epoch_acc
            model.save_pretrained(log_dir)
    writer.add_hparams(
        {
            "batch_per_gpu": opt.batch_per_gpu,
            "lr": opt.lr, 
            "epochs": opt.epochs,
            'num_of_gpu': len(opt.gpu.split(','))
        },
        {
            "best_acc": best_acc
        },
    )
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-per-gpu', type=int, help='batch size per gpu')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, help='epochs')
    parser.add_argument('--gpu', help='gpu id')
    opt = parser.parse_args()

    main(opt)

