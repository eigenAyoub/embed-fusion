from pathlib import Path
from typing import List, Dict, Optional
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import get_data
from loss import SimilarityLoss, Similarity, KLSimilarityLoss
from model import EncoderOnly 
from config import BATCH_SIZE as b_size

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-5
STEP_SIZE = 4
GAMMA = 0.1
PATIENCE = 5

class Trainer:
    """Handles model training, validation, and visualization
       Teacher data is optional. If teacher loaders are provided, teacher loss is computed.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        enc_config: Dict, # in / out
        dims,             # mrl dims
        now: str,         # checkpoints tag
        device: str = DEVICE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        teacher_train_loader: Optional[DataLoader] = None,
        teacher_val_loader: Optional[DataLoader] = None,
        teacher_loss_weight: float = 1.0,  # weight for teacher loss term
        checkpoint_dir: str = "checkpoints",
        save_freq: int = 5,  # Save every N iepochs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.teacher_train_loader = teacher_train_loader
        self.teacher_val_loader = teacher_val_loader
        self.device = device
        self.enc_config = enc_config
        self.now = now 
        self.mrl_dims = dims
        
        self.criterion = SimilarityLoss()
        self.teacher_loss_weight = teacher_loss_weight
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.best_val_loss = float('inf')
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def train_epoch(self, epoch: int) -> float:
        """Run one epoch of training.
        For each batch, the model processes the student_inputs once per MRL dimension.
        The student loss compares the output to student_inputs, while the teacher loss
        compares the same output to teacher_inputs (if provided).
        """
        self.model.train()
        running_loss = 0.0
        #self.teacher_loss_weight = 0.1 + (0.3*epoch)/29


        if self.teacher_train_loader is not None:
            loader = zip(self.train_loader, self.teacher_train_loader)
        else:
            loader = ((batch, None) for batch in self.train_loader)

        for student_batch, teacher_batch in loader:
            student_inputs = student_batch.to(self.device)
            self.optimizer.zero_grad()

            student_full= self.model(student_inputs)

            loss_student = 0.0
            loss_teacher = 0.0
            if teacher_batch is not None:
                teacher_targets = teacher_batch.to(self.device)

            for dim in self.mrl_dims:
                student_out = F.normalize(student_full[:,:dim], p=2, dim=1)
                loss_student += self.criterion(student_out, student_inputs)
                
                if teacher_batch is not None:
                    loss_teacher += self.criterion(student_out, teacher_targets)

            loss = 0.8 * loss_student + 0.2 * loss_teacher
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self) -> float:
        """Run validation using student data and teacher data if provided."""
        self.model.eval()
        running_loss = 0.0

        if self.teacher_val_loader is not None:
            loader = zip(self.val_loader, self.teacher_val_loader)
        else:
            loader = ((batch, None) for batch in self.val_loader)

        with torch.no_grad():
            for student_batch, teacher_batch in loader:
                student_inputs = student_batch.to(self.device)
                loss_student = 0.0
                loss_teacher = 0.0
                if teacher_batch is not None:
                    teacher_inputs = teacher_batch.to(self.device)
                
                student_full= self.model(student_inputs)

                for dim in self.mrl_dims:

                    student_out = F.normalize(student_full[:,:dim], p=2, dim=1)
                    loss_student += self.criterion(student_out, student_inputs)

                    if teacher_batch is not None:
                        loss_teacher += self.criterion(student_out, teacher_inputs)

                # Combine the losses; teacher loss is weighted.
                loss = 0.8 * loss_student + 0.2 * loss_teacher
                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        inDim  = self.enc_config["input_dim"]
        outDim = self.enc_config["output_dim"]

        path = self.checkpoint_dir / f'{inDim}_{outDim}_ep_{epoch:03d}_{self.now}.pth'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
            

    def train(self, num_epochs: int = NUM_EPOCHS):
        """Main training loop with checkpoint saving"""
        for epoch in range(num_epochs):

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.6f}')
            print(f'Val Loss: {val_loss:.6f}')
            
            # Save checkpoints
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch + 1)
            

def main():
  
    tag = "e5-small-no-ins"
    student_train_path = f"generate_data/{tag}/train_embeddings.npy"
    student_val_path = f"generate_data/{tag}/val_embeddings.npy"
    teacher_train_path = "generate_data/embeddings_data/f_mxbai_wiki_500k/train_embeddings.npy"
    teacher_val_path   = "generate_data/embeddings_data/f_mxbai_wiki_500k/val_embeddings.npy"

    train_loader = get_data(student_train_path)
    val_loader = get_data(student_val_path)
    
    teacher_train_loader = get_data(teacher_train_path)
    teacher_val_loader = get_data(teacher_val_path)

    model_config = {
                    'input_dim':  768,
                    'output_dim': 1024,
                }

    inDim = model_config["input_dim"]
    outDim = model_config["output_dim"]

    COMPRESSED_DIMENSIONS = [32, 64, 100, 150, 200, 250, 300, 350, 384, 512, 768, outDim]

    model = EncoderOnly(model_config)
    now = datetime.datetime.now().strftime("%H%M%S")
    
    log_line = f"run {now} {inDim} {outDim} {COMPRESSED_DIMENSIONS} Loader batch size {b_size} Obs: first teacher ish thing with normal loss run\n"

    with open("logs.txt", "a") as f:
        f.write(log_line)

    trainer = Trainer(
        model, 
        train_loader, 
        val_loader,
        enc_config=model_config,
        dims=COMPRESSED_DIMENSIONS,
        now=now,
        checkpoint_dir=f"models_pth/{inDim}_{outDim}",
        save_freq=5,
        teacher_train_loader=teacher_train_loader,   # set to None to disable
        teacher_val_loader=teacher_val_loader,         # set to None to disable
        teacher_loss_weight=0.3  # adjust this weight as needed
    )
    
    trainer.train()

if __name__ == "__main__":
    main()
