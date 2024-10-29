import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import TextDataset
from utils.model import TextClsModel
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score, recall_score, f1_score
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, total_loss / len(data_loader), recall, f1

def train():
    # 1.配置必要参数(路径需要按照实际路径修改）
    pretrain_model = r"./bert-base-uncased"
    current_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    log_dir = f'runs/text_classification_{current_time}'
    df = pd.read_csv(r"./datasets/events_set1/italy_earthquake_aug_2016/italy_earthquake_aug_2016_train.tsv", sep='\t')
    train_file_path = r"./datasets/events_set1/italy_earthquake_aug_2016/italy_earthquake_aug_2016_test.tsv"
    test_file_path = r"./datasets/events_set1/italy_earthquake_aug_2016/italy_earthquake_aug_2016_train.tsv"

    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
    MAX_LEN = 256 # 最长的句子长度
    BATCH_SIZE = 16 # 批次大小
    NUM_CLASS = 11 #类别数量
    EPOCHS = 6 # 迭代次数
    # 2. 创建数据加载器
    train_dataset = TextDataset(train_file_path, tokenizer, MAX_LEN)
    val_dataset = TextDataset(test_file_path, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 3. 加载模型并移动到GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = TextClsModel(n_classes=NUM_CLASS,pretrain_model=pretrain_model)
    model = model.to(device)

    # 6. 设置优化器和学习率调度器

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    global_f1 = 0
    # 7. 记录训练指标
    writer = SummaryWriter(log_dir)
    # 7. 训练和验证模型
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_accuracy, val_loss, val_recall, val_f1 = eval_model(model, val_loader, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
        writer.add_scalar('Recall/validation', val_recall, epoch)
        writer.add_scalar('F1/validation', val_f1, epoch)
        if val_f1 > global_f1:
            global_f1 = val_f1
            torch.save(model.state_dict(),f"{log_dir}/epoch_{epoch}_model_acc{val_f1:6f}")
            print(f"Model saved at epoch {epoch} with validation f1 {val_f1:.6f}")

        print(f'Train loss {train_loss:4f} Validation loss {val_loss:4f} Validation accuracy {val_accuracy:4f} Validation recall {val_recall:4f} Validation F1 {val_f1:4f}')

if __name__ == "__main__":
    train()
