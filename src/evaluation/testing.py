import argparse
import random
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer
from src.utils.config import Config
from src.model.model import TripletLiteraryStyleNetwork
from src.data.dataset import LiteraryStyleDataset
from src.model.model_utils import load_model, get_model_path

def load_trained_model(device):
    model = TripletLiteraryStyleNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    _, _ = load_model(model, optimizer, get_model_path())
    model.eval()
    return model

def tokenize_text(text, tokenizer, device):
    return tokenizer(text, padding='max_length', truncation=True, max_length=Config.MAX_LENGTH, return_tensors='pt').to(device)

def calculate_similarity(model, text1, text2, tokenizer, device):
    with torch.no_grad():
        inputs1 = tokenize_text(text1, tokenizer, device)
        inputs2 = tokenize_text(text2, tokenizer, device)
        
        embedding1 = model.forward_one(inputs1['input_ids'], inputs1['attention_mask'])
        embedding2 = model.forward_one(inputs2['input_ids'], inputs2['attention_mask'])
        
        similarity = cosine_similarity(embedding1, embedding2)
    return similarity.item()

def display_random_triplet_similarities(model, dataset, tokenizer, device):
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    random_index = random.randint(0, len(dataset) - 1)
    triplet = dataset[random_index]

    if isinstance(triplet, dict):  # For tokenized datasets
        anchor_text = tokenizer.decode(triplet['anchor_ids'], skip_special_tokens=True)
        positive_text = tokenizer.decode(triplet['positive_ids'], skip_special_tokens=True)
        negative_text = tokenizer.decode(triplet['negative_ids'], skip_special_tokens=True)
    elif isinstance(triplet, (list, tuple)) and len(triplet) == 3:  # For non-tokenized datasets
        anchor_text, positive_text, negative_text = triplet
    else:
        print("Error: Unexpected triplet format.")
        return

    anchor_positive_sim = calculate_similarity(model, anchor_text, positive_text, tokenizer, device)
    anchor_negative_sim = calculate_similarity(model, anchor_text, negative_text, tokenizer, device)

    print("\nRandom Triplet Analysis:")
    print("\nAnchor Text:")
    print(anchor_text[:200] + "..." if len(anchor_text) > 200 else anchor_text)
    
    print("\nPositive (Similar) Text:")
    print(positive_text[:200] + "..." if len(positive_text) > 200 else positive_text)
    print(f"Similarity score with anchor: {anchor_positive_sim:.4f}")
    
    print("\nNegative (Different) Text:")
    print(negative_text[:200] + "..." if len(negative_text) > 200 else negative_text)
    print(f"Similarity score with anchor: {anchor_negative_sim:.4f}")

def compare_custom_texts(model, text1, text2, tokenizer, device):
    similarity = calculate_similarity(model, text1, text2, tokenizer, device)
    
    print("\nCustom Text Comparison:")
    print("\nText 1:")
    print(text1[:200] + "..." if len(text1) > 200 else text1)
    print("\nText 2:")
    print(text2[:200] + "..." if len(text2) > 200 else text2)
    print(f"\nSimilarity score: {similarity:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test the Literary Style Analysis model")
    parser.add_argument("--random", action="store_true", help="Analyze a random triplet from the dataset")
    parser.add_argument("--text1", type=str, help="First text for custom comparison")
    parser.add_argument("--text2", type=str, help="Second text for custom comparison")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_trained_model(device)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = LiteraryStyleDataset(Config.TOKENIZED_TRIPLETS_FILE, tokenized=True)

    if args.random:
        display_random_triplet_similarities(model, dataset, tokenizer, device)
    elif args.text1 and args.text2:
        compare_custom_texts(model, args.text1, args.text2, tokenizer, device)
    else:
        print("Please specify either --random for random triplet analysis or both --text1 and --text2 for custom text comparison.")

if __name__ == "__main__":
    main()