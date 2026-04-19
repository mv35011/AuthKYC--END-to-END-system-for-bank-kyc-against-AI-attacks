import cv2
import torch
import os
import glob
import random
from facenet_pytorch import MTCNN
from torchvision.transforms import v2 as T


def save_augmented_variants(source_path, target_dir, target_count):
    """Creates diverse variants of a tensor to reach physical balance."""
    augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.4),
    ])

    base_tensor = torch.load(source_path, weights_only=False)  # [N, T, C, H, W]
    existing = len(glob.glob(f"{target_dir}/custom_*.pt"))
    copy_idx = existing

    while len(glob.glob(f"{target_dir}/custom_*.pt")) < target_count:
        aug_sequences = []
        for seq in base_tensor:
            aug_sequences.append(augment(seq))

        dst = os.path.join(target_dir, f"custom_aug_{copy_idx}.pt")
        torch.save(torch.stack(aug_sequences), dst)
        copy_idx += 1


class DeepfakeDataExtractor:
    def __init__(self, device='cuda', image_size=224, seq_length=16, max_sequences=8):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.seq_length = seq_length
        self.max_sequences = max_sequences
        self.mtcnn = MTCNN(image_size=self.image_size, margin=40, keep_all=False, post_process=False,
                           device=self.device)

        # FIX: We only convert to float (0.0 to 1.0). Normalization happens in dataset.py
        self.transform = T.Compose([T.ConvertImageDtype(torch.float32)])

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def process_video(self, video_path, output_dir, prefix=""):
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{prefix}{video_name}.pt")
        if os.path.exists(output_path): return True

        frames = self.extract_frames(video_path)
        if not frames: return False

        valid_faces = []
        max_frames_needed = self.seq_length * self.max_sequences

        for i in range(0, len(frames), 32):
            chunk = frames[i:i + 32]
            try:
                faces = self.mtcnn(chunk)
                if faces is not None:
                    for face in faces:
                        if face is not None:
                            valid_faces.append(self.transform(face / 255.0))
            except Exception:
                continue
            if len(valid_faces) >= max_frames_needed: break

        if len(valid_faces) < self.seq_length: return False

        valid_faces = valid_faces[:max_frames_needed]
        face_tensor = torch.stack(valid_faces)
        num_sequences = len(face_tensor) // self.seq_length
        exact_frames_needed = num_sequences * self.seq_length

        sequences = face_tensor[:exact_frames_needed].view(num_sequences, self.seq_length, 3, self.image_size,
                                                           self.image_size)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"Processed {prefix}{video_name} -> {num_sequences} seqs")
        return True


if __name__ == "__main__":
    extractor = DeepfakeDataExtractor()
    train_real_dir = './processed_tensors/train/real'
    train_fake_dir = './processed_tensors/train/fake'
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_fake_dir, exist_ok=True)

    # 1. Custom Anchors (Extract & Augment to 250)
    custom_paths = glob.glob('/workspace/custom_webcam/*.mp4')
    extracted_custom_files = []
    for path in custom_paths:
        if extractor.process_video(path, train_real_dir, prefix="custom_"):
            video_name = os.path.basename(path).split('.')[0]
            extracted_custom_files.append(os.path.join(train_real_dir, f"custom_{video_name}.pt"))

    target_custom = 250
    current = len(extracted_custom_files)
    if 0 < current < target_custom:
        print(f"Generating diverse variants to reach {target_custom}...")
        idx = 0
        while len(glob.glob(f"{train_real_dir}/custom_*.pt")) < target_custom:
            save_augmented_variants(extracted_custom_files[idx % current], train_real_dir, target_custom)
            idx += 1

    # 2. FF++ Real (Cap 250)
    real_paths = glob.glob('/workspace/ff-c23/FaceForensics++_C23/original/*.mp4')
    random.shuffle(real_paths)
    count = 0
    for path in real_paths:
        if extractor.process_video(path, train_real_dir): count += 1
        if count >= 250: break

    # 3. FF++ Fake (Cap 500)
    fake_paths = []
    for folder in ['Deepfakes', 'Face2Face']:
        fake_paths.extend(glob.glob(f"/workspace/ff-c23/FaceForensics++_C23/{folder}/*.mp4"))
    random.shuffle(fake_paths)
    count = 0
    for path in fake_paths:
        if extractor.process_video(path, train_fake_dir): count += 1
        if count >= 500: break