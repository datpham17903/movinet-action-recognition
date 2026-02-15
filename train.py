"""
Fine-tuning script for R3D Movinet Action Recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
import glob
import random


class VideoDataset(Dataset):
    """Dataset for video classification"""
    
    def __init__(self, data_dir: str, num_frames: int = 16, image_size: int = 224,
                 transform=None, is_train: bool = True):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_train = is_train
        self.transform = transform or self._default_transform()
        
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            for video_file in class_dir.glob('*.mp4'):
                self.samples.append((str(video_file), self.class_to_idx[class_name]))
            
            for video_file in class_dir.glob('*.avi'):
                self.samples.append((str(video_file), self.class_to_idx[class_name]))
    
    def _default_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)
                frames.append(frame)
            else:
                frames.append(torch.zeros(3, self.image_size, self.image_size))
        
        cap.release()
        
        while len(frames) < self.num_frames:
            frames.append(torch.zeros(3, self.image_size, self.image_size))
        
        video = torch.stack(frames[:self.num_frames])
        return video
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        try:
            video = self._load_video(video_path)
        except:
            video = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)
        
        video = video.permute(1, 0, 2, 3)
        
        return video, label


def create_model(num_classes: int, pretrained: bool = True):
    """Create R3D model for fine-tuning"""
    from torchvision.models.video import r3d_18, R3D_18_Weights
    
    if pretrained:
        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    else:
        model = r3d_18(weights=None)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def train_model(model, train_loader, device, num_epochs=10, lr=0.001):
    """Train the model"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%')
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%')
    
    return model


def evaluate_model(model, val_loader, device):
    """Evaluate the model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def prepare_sample_data():
    """Prepare sample data structure for testing"""
    dataset_dir = Path('dataset/train')
    
    sample_classes = ['walking', 'running', 'dancing']
    
    for cls in sample_classes:
        cls_dir = dataset_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        print(f'Created: {cls_dir}')
    
    print('\nDataset structure created!')
    print('To train: put .mp4 videos in dataset/train/<action_name>/')
    print('Then run: python train.py --data_dir dataset/train --epochs 10')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune R3D for action recognition')
    parser.add_argument('--data_dir', type=str, default='dataset/train',
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames per video')
    parser.add_argument('--save_path', type=str, default='models/fine_tuned.pth',
                       help='Path to save model')
    parser.add_argument('--prepare_data', action='store_true',
                       help='Just prepare data structure')
    
    args = parser.parse_args()
    
    if args.prepare_data:
        prepare_sample_data()
    else:
        print('Loading dataset...')
        dataset = VideoDataset(args.data_dir, num_frames=args.num_frames)
        
        if len(dataset) == 0:
            print('No data found! Run with --prepare_data first')
            exit(1)
        
        print(f'Found {len(dataset.classes)} classes: {dataset.classes}')
        print(f'Found {len(dataset)} samples')
        
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        device = torch.device('cpu')
        print(f'Using device: {device} (forced CPU due to GPU compatibility)')
        
        model = create_model(num_classes=len(dataset.classes))
        model = model.to(device)
        
        print('Starting training...')
        model = train_model(model, train_loader, device, num_epochs=args.epochs, lr=args.lr)
        
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': dataset.classes,
            'class_to_idx': dataset.class_to_idx,
        }, args.save_path)
        
        print(f'Model saved to {args.save_path}')
