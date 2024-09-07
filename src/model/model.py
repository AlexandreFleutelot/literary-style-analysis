import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from src.utils.config import Config
from src.utils.logger import logger

class TripletLiteraryStyleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        
        # Freeze the embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze the specified number of layers
        total_layers = len(self.bert.encoder.layer)
        layers_to_freeze = min(Config.NUM_FROZEN_LAYERS, total_layers)
        logger.info(f"Freezing first {layers_to_freeze} layers out of {total_layers}")
        for layer in self.bert.encoder.layer[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
    def forward_one(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        return self.projection(pooled_output)
    
    def forward(self, anchor_ids, anchor_mask, positive_ids, positive_mask, negative_ids, negative_mask):
        anchor_embed = self.forward_one(anchor_ids, anchor_mask)
        positive_embed = self.forward_one(positive_ids, positive_mask)
        negative_embed = self.forward_one(negative_ids, negative_mask)
        return anchor_embed, positive_embed, negative_embed

def triplet_loss(anchor, positive, negative, margin=Config.MARGIN):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + margin)
    return losses.mean()

def calculate_similarity(model, text1, text2, tokenizer, device):
    model.eval()
    with torch.no_grad():
        inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_LENGTH).to(device)
        inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_LENGTH).to(device)
        
        embedding1 = model.forward_one(inputs1['input_ids'], inputs1['attention_mask'])
        embedding2 = model.forward_one(inputs2['input_ids'], inputs2['attention_mask'])
        
        similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()