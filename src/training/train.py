import torch
from torch.utils.data import DataLoader
from src.utils.config import Config
from src.model.model import triplet_loss, calculate_similarity
from src.utils.logger import logger
from src.model.model_utils import save_model, get_model_path
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm

def train_triplet(model, train_loader, optimizer, num_epochs, device):
    model.train()
    scaler = GradScaler()
    epoch=0
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        try:
            for batch in progress_bar:
                anchor_ids = batch['anchor_ids'].to(device)
                anchor_mask = batch['anchor_mask'].to(device)
                positive_ids = batch['positive_ids'].to(device)
                positive_mask = batch['positive_mask'].to(device)
                negative_ids = batch['negative_ids'].to(device)
                negative_mask = batch['negative_mask'].to(device)

                with autocast(device_type='cuda', dtype=torch.float16):
                    anchor_embed, positive_embed, negative_embed = model(
                        anchor_ids, anchor_mask, 
                        positive_ids, positive_mask, 
                        negative_ids, negative_mask
                    )
                    
                    loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                running_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            epoch_loss = running_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
            if Config.SAVE_INTERMEDIATE_MODELS and (epoch + 1) % Config.SAVE_INTERVAL == 0:
                save_model(model, optimizer, epoch + 1, epoch_loss, get_model_path(epoch + 1))
        
        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            logger.exception("Stack trace:")
            break  # Exit the training loop if an exception occurs
    
    # Save the final model
    
    final_epoch = epoch + 1
    final_loss = epoch_loss if 'epoch_loss' in locals() else None
    save_model(model, optimizer, final_epoch, final_loss, get_model_path())
    logger.info(f"Training completed. Final model saved after {final_epoch} epochs.")

def test_triplet(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            anchor_ids = batch['anchor_ids'].to(device)
            anchor_mask = batch['anchor_mask'].to(device)
            positive_ids = batch['positive_ids'].to(device)
            positive_mask = batch['positive_mask'].to(device)
            negative_ids = batch['negative_ids'].to(device)
            negative_mask = batch['negative_mask'].to(device)

            anchor_embed, positive_embed, negative_embed = model(
                anchor_ids, anchor_mask, 
                positive_ids, positive_mask, 
                negative_ids, negative_mask
            )
            
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            total_loss += loss.item()

            # Calculate similarities
            pos_similarity = calculate_similarity(anchor_embed, positive_embed)
            neg_similarity = calculate_similarity(anchor_embed, negative_embed)

            # Count correct predictions (positive should be more similar than negative)
            correct_predictions += torch.sum(pos_similarity > neg_similarity).item()
            total_predictions += anchor_embed.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions

    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy

def calculate_similarity(embed1, embed2):
    return torch.nn.functional.cosine_similarity(embed1, embed2)

def create_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)