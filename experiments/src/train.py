"""
Training script for CodeFault model.

Usage:
    python train.py --config configs/base.yaml
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml

from models.codefault import CodeFault, ContrastiveLoss
from data.dataset import CodeFaultDataset, create_dataloaders
from utils.metrics import compute_metrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for CodeFault model."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = CodeFault(config.get('model', {})).to(self.device)
        
        # Initialize tokenizer
        code_model = config.get('model', {}).get('code_model', 'Salesforce/codet5p-770m')
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_model, trust_remote_code=True)
        
        # Load log vocabulary
        log_vocab_path = config.get('data', {}).get('log_vocab', 'data/log_vocab.json')
        if os.path.exists(log_vocab_path):
            with open(log_vocab_path) as f:
                self.log_vocab = json.load(f)
        else:
            self.log_vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # Initialize dataloaders
        data_path = config.get('data', {}).get('path', 'data')
        batch_size = config.get('training', {}).get('batch_size', 32)
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_path, self.code_tokenizer, self.log_vocab, batch_size
        )
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.get('training', {}).get('temperature', 0.07)
        )
        self.lambda_contrastive = config.get('training', {}).get('lambda_contrastive', 0.1)
        
        # Optimizer
        lr = config.get('training', {}).get('learning_rate', 2e-5)
        weight_decay = config.get('training', {}).get('weight_decay', 0.01)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        num_epochs = config.get('training', {}).get('num_epochs', 10)
        warmup_epochs = config.get('training', {}).get('warmup_epochs', 1)
        
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs - warmup_epochs)
        self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # Training state
        self.global_step = 0
        self.best_auc = 0.0
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in pbar:
            # Move to device
            code_input_ids = batch['code_input_ids'].to(self.device)
            code_attention_mask = batch['code_attention_mask'].to(self.device)
            log_input_ids = batch['log_input_ids'].to(self.device)
            log_attention_mask = batch['log_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                code_input_ids, code_attention_mask,
                log_input_ids, log_attention_mask
            )
            
            # Compute losses
            bce = self.bce_loss(outputs['logits'], labels)
            contrastive = self.contrastive_loss(outputs['fused_embedding'], labels)
            loss = bce + self.lambda_contrastive * contrastive
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            self.global_step += 1
            
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.writer.add_scalar('train/bce_loss', bce.item(), self.global_step)
            self.writer.add_scalar('train/contrastive_loss', contrastive.item(), self.global_step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self, loader, split: str = 'val') -> dict:
        """Evaluate model on given loader."""
        self.model.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0.0
        
        for batch in tqdm(loader, desc=f'Evaluating {split}'):
            code_input_ids = batch['code_input_ids'].to(self.device)
            code_attention_mask = batch['code_attention_mask'].to(self.device)
            log_input_ids = batch['log_input_ids'].to(self.device)
            log_attention_mask = batch['log_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(
                code_input_ids, code_attention_mask,
                log_input_ids, log_attention_mask
            )
            
            loss = self.bce_loss(outputs['logits'], labels)
            total_loss += loss.item()
            
            all_probs.extend(outputs['probability'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        # Compute metrics
        metrics = compute_metrics(all_labels, all_probs)
        metrics['loss'] = total_loss / len(loader)
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f'{split}/{key}', value, self.global_step)
            
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
            logger.info(f'New best model saved with AUC: {metrics["auc"]:.4f}')
    
    def train(self):
        """Full training loop."""
        num_epochs = self.config.get('training', {}).get('num_epochs', 10)
        
        logger.info(f'Starting training for {num_epochs} epochs')
        logger.info(f'Device: {self.device}')
        logger.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f}')
            
            # Evaluate
            val_metrics = self.evaluate(self.val_loader, 'val')
            logger.info(f'Epoch {epoch} - Val AUC: {val_metrics["auc"]:.4f}, F1: {val_metrics["f1"]:.4f}')
            
            # Save checkpoint
            is_best = val_metrics['auc'] > self.best_auc
            if is_best:
                self.best_auc = val_metrics['auc']
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Step scheduler
            self.scheduler.step()
        
        # Final test evaluation
        logger.info('Evaluating on test set...')
        test_metrics = self.evaluate(self.test_loader, 'test')
        logger.info(f'Test Results - AUC: {test_metrics["auc"]:.4f}, F1: {test_metrics["f1"]:.4f}')
        
        # Save final results
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        self.writer.close()
        return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Add timestamp to output dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = f"outputs/{timestamp}"
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
