import torch
from transformers import BertTokenizer
from utils.model import TextClsModel
from utils.dataset import type2id
class Inference:
    def __init__(self, model_path, pretrain_model_path, num_classes):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
        self.model = TextClsModel(n_classes=num_classes, pretrain_model=pretrain_model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, text, max_len=256):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def predict(self, text):
        inputs = self.preprocess(text)
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            _, preds = torch.max(outputs, dim=1)
        return preds.item()

if __name__ == "__main__":
    # 加载模型和分词器
    model_path = '/Users/williamlu/Downloads/pythonProject/runs/text_classification_2024-07-05_06-12-03/epoch_5_model_acc0.765598'  # 替换为实际的模型文件路径
    pretrain_model_path = r"./bert-base-uncased"
    num_classes = 8
    id2type = {v:k for k,v in type2id.items()}
    # 初始化推理对象
    inference = Inference(model_path, pretrain_model_path, num_classes)

    # 用户输入文本
    while True:
        user_input = input("请输入文本: ")
        prediction = inference.predict(user_input)
        print(f"预测的类别: {id2type[prediction+1]}")
