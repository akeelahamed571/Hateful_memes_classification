import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class HatefulMemesFusionModel(nn.Module):
    def __init__(self, text_hidden_dim=768, image_output_dim=128, fusion_output_dim=256):
        super(HatefulMemesFusionModel, self).__init__()
        
        # Text Encoder: BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # Image Encoder: simple CNN
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.flatten = nn.Flatten()
        self.image_fc = nn.Linear(64 * 4 * 4, image_output_dim)

        # Fusion Layer
        self.fusion_fc = nn.Linear(text_hidden_dim + image_output_dim, fusion_output_dim)

        # Classifier
        self.classifier = nn.Linear(fusion_output_dim, 2)  # Binary classification

    def forward(self, input_ids, attention_mask, image):
        # Text features from BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output  # shape: (batch_size, 768)

        # Image features
        image_feat = self.image_encoder(image)
        image_feat = self.flatten(image_feat)
        image_features = self.image_fc(image_feat)  # shape: (batch_size, 128)

        # Fusion
        fused = torch.cat((text_features, image_features), dim=1)
        fusion_output = F.relu(self.fusion_fc(fused))

        # Classification
        output = self.classifier(fusion_output)
        return output


def train_torch():
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return train


def test_torch():
    def test(model, test_loader, criterion, device):
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                image = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, image=image)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total
    return test
