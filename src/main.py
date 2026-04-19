import os
import torch
import config
from dataloader import get_dataloaders
from model import build_model
from train import train

if __name__ == '__main__':
    BATCH_SIZE = config.BATCH_SIZE
    K = 4 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE, k=K)

    model = build_model()

    resume_path = os.path.join(config.CHECKPOINT_DIR, config.LAST_CKPT_NAME)
    resume_path = resume_path if os.path.exists(resume_path) else None

    print('Training Started..')
    train(model, train_loader, val_loader, device, resume_path=resume_path)