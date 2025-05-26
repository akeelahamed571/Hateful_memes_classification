import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO


class HatefulMemesDataset(Dataset):
    def __init__(self, split="train", max_length=128):
        print(f"📥 Loading Hateful Memes split: {split}")
        self.dataset = load_dataset("neuralcatcher/hateful_memes", split=split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.max_length = max_length
        print(f"✅ Loaded {len(self.dataset)} samples from {split} split")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if idx == 0:
            print("Dataset columns:", list(item.keys()))
        # Process text
        text = item["text"]
        text_tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        
        # Process image
        image_tensor = torch.zeros((3, 224, 224))  # default image if anything goes wrong
        try:
            image_info = item.get("img", None)
            if image_info and isinstance(image_info, dict) and "url" in image_info:
                img_url = image_info["url"]
                img_bytes = requests.get(img_url, timeout=5).content
                image = Image.open(BytesIO(img_bytes)).convert("RGB")
                image_tensor = self.transform(image)
            else:
                print(f"⚠️ Missing image or URL in item at index {idx}")
        except Exception as e:
            print(f"❌ Failed to load image for item {idx}: {e}")


        label = torch.tensor(item["label"], dtype=torch.long)

        return {
            "input_ids": text_tokens["input_ids"].squeeze(0),
            "attention_mask": text_tokens["attention_mask"].squeeze(0),
            "image": image_tensor,
            "label": label
        }


def load_partition(batch_size=32):
    print("🧪 Preparing data loaders for Hateful Memes...")
    train_data = HatefulMemesDataset(split="train")
    val_data = HatefulMemesDataset(split="validation")
    test_data = HatefulMemesDataset(split="test")

    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=True,
    )
    #val_loader = DataLoader(val_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    #test_loader = DataLoader(test_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0, pin_memory=True)
    print("✅ Data loaders ready.")
    return train_loader, val_loader, test_loader


def gl_model_torch_validation(batch_size):
    _, val_loader, _ = load_partition(batch_size=batch_size)
    return val_loader
