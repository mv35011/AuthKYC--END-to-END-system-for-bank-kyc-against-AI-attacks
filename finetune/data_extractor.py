import cv2
import torch
import os
import glob
import random
import shutil
from facenet_pytorch import MTCNN
from torchvision import transforms


class DeepfakeDataExtractor:
    def __init__(self, device='cuda', image_size=224, seq_length=16, max_sequences=8):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.seq_length = seq_length
        self.max_sequences = max_sequences

        self.mtcnn = MTCNN(
            image_size=self.image_size, margin=40, keep_all=False,
            post_process=False, device=self.device
        )
        self.transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames

    def process_video(self, video_path, output_dir, prefix=""):
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{prefix}{video_name}.pt")

        if os.path.exists(output_path): return True  # Return True if successful

        frames = self.extract_frames(video_path)
        if not frames: return False

        valid_faces = []
        batch_size = 32
        max_frames_needed = self.seq_length * self.max_sequences

        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            try:
                faces = self.mtcnn(chunk)
                if faces is not None:
                    for face in faces:
                        if face is not None:
                            valid_faces.append(self.transform(face / 255.0))
            except Exception as e:
                continue

            if len(valid_faces) >= max_frames_needed: break

        if len(valid_faces) < self.seq_length: return False

        valid_faces = valid_faces[:max_frames_needed]
        face_tensor = torch.stack(valid_faces)
        num_sequences = len(face_tensor) // self.seq_length

        exact_frames_needed = num_sequences * self.seq_length
        face_tensor = face_tensor[:exact_frames_needed]

        sequences = face_tensor.view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )

        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"Processed {prefix}{video_name} -> {num_sequences} seqs")
        return True


if __name__ == "__main__":
    extractor = DeepfakeDataExtractor(seq_length=16, image_size=224, max_sequences=8)
    print("[System] Initializing Domain Adaptation Extractor (500/500 Target)...")

    train_real_dir = './processed_tensors/train/real'
    train_fake_dir = './processed_tensors/train/fake'
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_fake_dir, exist_ok=True)

    # --- 1. CUSTOM WEBCAM VIDEOS (Extract & Auto-Duplicate to 250) ---
    custom_paths = glob.glob('/workspace/custom_webcam/*.mp4')
    print(f"Found {len(custom_paths)} Custom Anchor Videos.")

    extracted_custom_files = []
    for path in custom_paths:
        if extractor.process_video(path, train_real_dir, prefix="custom_"):
            video_name = os.path.basename(path).split('.')[0]
            extracted_custom_files.append(os.path.join(train_real_dir, f"custom_{video_name}.pt"))

    # Auto-Duplicate logic
    target_custom_count = 250
    current_count = len(extracted_custom_files)
    if current_count > 0 and current_count < target_custom_count:
        print(f"Duplicating custom data to reach {target_custom_count} files...")
        idx = 0
        while len(glob.glob(f"{train_real_dir}/custom_*.pt")) < target_custom_count:
            src = extracted_custom_files[idx % current_count]
            dst = src.replace('.pt', f"_copy{len(glob.glob(f'{train_real_dir}/custom_*.pt'))}.pt")
            shutil.copy(src, dst)
            idx += 1
    print(f"Custom Anchor Data physically balanced to: {len(glob.glob(f'{train_real_dir}/custom_*.pt'))} files.")

    # --- 2. FF++ REAL VIDEOS (Cap at 250) ---
    real_paths = glob.glob('/workspace/ff-c23/FaceForensics++_C23/original/*.mp4')
    random.seed(42)
    random.shuffle(real_paths)

    print("\n--- Extracting FF++ Real Data (Target: 250) ---")
    ff_real_count = 0
    for path in real_paths:
        if extractor.process_video(path, train_real_dir):
            ff_real_count += 1
        if ff_real_count >= 250: break

    # --- 3. FF++ FAKE VIDEOS (Cap at 500) ---
    fake_paths = []
    fake_folders = [
        '/workspace/ff-c23/FaceForensics++_C23/Deepfakes',
        '/workspace/ff-c23/FaceForensics++_C23/Face2Face'
    ]
    for folder in fake_folders: fake_paths.extend(glob.glob(f"{folder}/*.mp4"))
    random.shuffle(fake_paths)

    print("\n--- Extracting FF++ Fake Data (Target: 500) ---")
    ff_fake_count = 0
    for path in fake_paths:
        if extractor.process_video(path, train_fake_dir):
            ff_fake_count += 1
        if ff_fake_count >= 500: break

    print("\n[System] Extraction & Physical Balancing Complete.")
    print(f"Total REAL Tensors: {len(glob.glob(f'{train_real_dir}/*.pt'))}")
    print(f"Total FAKE Tensors: {len(glob.glob(f'{train_fake_dir}/*.pt'))}")