"""
Demo training script - creates synthetic data and fine-tunes the model
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random


def create_synthetic_training_data():
    """Create synthetic training data from sample video"""
    
    dataset_dir = Path('dataset/train')
    classes = ['walking', 'running', 'dancing']
    
    for cls in classes:
        (dataset_dir / cls).mkdir(parents=True, exist_ok=True)
    
    sample_video = 'sample_video.mp4'
    if not os.path.exists(sample_video):
        print(f"Error: {sample_video} not found!")
        return False
    
    cap = cv2.VideoCapture(sample_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        print("Error: Could not read frames from video")
        return False
    
    for cls_idx, cls_name in enumerate(classes):
        print(f"Creating training samples for: {cls_name}")
        
        for sample_idx in range(10):
            start_frame = random.randint(0, max(0, len(frames) - 32))
            clip_frames = frames[start_frame:start_frame + 32]
            
            if len(clip_frames) < 16:
                clip_frames = clip_frames + [clip_frames[-1]] * (16 - len(clip_frames))
            
            output_path = dataset_dir / cls_name / f'{cls_name}_{sample_idx:03d}.avi'
            
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (224, 224))
            
            for frame in clip_frames[:32]:
                frame_resized = cv2.resize(frame, (224, 224))
                
                if cls_name == 'running':
                    frame_resized = cv2.convertScaleAbs(frame_resized, alpha=1.2, beta=10)
                elif cls_name == 'dancing':
                    frame_resized = cv2.convertScaleAbs(frame_resized, alpha=1.1, beta=5)
                    frame_resized = cv2.GaussianBlur(frame_resized, (5, 5), 0)
                elif cls_name == 'walking':
                    frame_resized = cv2.convertScaleAbs(frame_resized, alpha=0.9, beta=0)
                
                out.write(frame_resized)
            
            out.release()
    
    print(f"\nCreated training data in {dataset_dir}/")
    for cls in classes:
        count = len(list((dataset_dir / cls).glob('*.mp4')))
        print(f"  {cls}: {count} videos")
    
    return True


def count_samples():
    """Count available training samples"""
    dataset_dir = Path('dataset/train')
    classes = ['walking', 'running', 'dancing']
    
    total = 0
    for cls in classes:
        count = len(list((dataset_dir / cls).glob('*.mp4')))
        print(f"  {cls}: {count}")
        total += count
    
    return total


if __name__ == '__main__':
    print("=" * 50)
    print("Creating Synthetic Training Data")
    print("=" * 50)
    
    success = create_synthetic_training_data()
    
    if success:
        print("\n" + "=" * 50)
        print("Training Data Summary")
        print("=" * 50)
        total = count_samples()
        print(f"\nTotal samples: {total}")
        print("\nTo train the model, run:")
        print("  python train.py --data_dir dataset/train --epochs 5 --batch_size 2")
