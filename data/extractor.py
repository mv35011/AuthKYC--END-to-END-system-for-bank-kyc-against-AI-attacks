import cv2
import torch
import os
import glob
import random
from facenet_pytorch import MTCNN
from torchvision import transforms

class DeepfakeDataExtractor:
    # --- ADDED MAX_SEQUENCES TO INIT ---
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

    def process_video(self, video_path, output_dir):
        video_name = os.path.basename(video_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{video_name}.pt")

        if os.path.exists(output_path): return

        frames = self.extract_frames(video_path)
        if not frames: return

        valid_faces = []
        batch_size = 32

        # --- THE DYNAMIC LIMIT ---
        # Now it listens to the class setting (8) instead of a hardcoded 2
        max_frames_needed = self.seq_length * self.max_sequences
        # -------------------------

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

            # --- EARLY STOPPING ---
            if len(valid_faces) >= max_frames_needed:
                break

        if len(valid_faces) < self.seq_length: return

        valid_faces = valid_faces[:max_frames_needed]

        face_tensor = torch.stack(valid_faces)
        num_sequences = len(face_tensor) // self.seq_length

        sequences = face_tensor.view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )

        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"Processed {video_name} -> {num_sequences} seqs saved to {output_dir.split('/')[-2]}")


if __name__ == "__main__":
    # --- 1. INITIALIZE EXTRACTOR ---
    # Setting max_sequences directly in the constructor
    extractor = DeepfakeDataExtractor(seq_length=16, image_size=224, max_sequences=8)

    print("[System] Initializing Data Aggregator (Experiment 2)...")

    # --- 2. GATHER REAL VIDEOS (FF++ and Celeb-DF) ---
    real_paths = []
    real_folders = [
        '/workspace/ff-c23/FaceForensics++_C23/original',
        '/workspace/celeb-df-v2/Celeb-real',
        '/workspace/celeb-df-v2/YouTube-real'
    ]
    for folder in real_folders:
        real_paths.extend(glob.glob(f"{folder}/*.mp4"))

    # --- 3. GATHER FAKE VIDEOS (FF++) ---
    fake_paths = []
    fake_folders = [
        '/workspace/ff-c23/FaceForensics++_C23/Deepfakes',
        '/workspace/ff-c23/FaceForensics++_C23/Face2Face',
        '/workspace/ff-c23/FaceForensics++_C23/FaceSwap',
        '/workspace/ff-c23/FaceForensics++_C23/FaceShifter',
        '/workspace/ff-c23/FaceForensics++_C23/DeepFakeDetection'
    ]
    for folder in fake_folders:
        fake_paths.extend(glob.glob(f"{folder}/*.mp4"))

    # --- 4. SHUFFLE AND CAP AT 1500 (The Balance Fix) ---
    random.seed(42)
    random.shuffle(real_paths)
    random.shuffle(fake_paths)

    real_videos = real_paths[:1500]
    fake_videos = fake_paths[:1500]

    print(f"Total Real Videos Selected: {len(real_videos)}")
    print(f"Total Fake Videos Selected: {len(fake_videos)}")

    # --- 5. STRICT TRAIN/VAL SPLIT (80/20) ---
    train_real = real_videos[:1200]
    val_real = real_videos[1200:]

    train_fake = fake_videos[:1200]
    val_fake = fake_videos[1200:]

    # --- 6. EXECUTE EXTRACTION ---
    print("\n--- Extracting Training Data (REAL) ---")
    for path in train_real:
        extractor.process_video(path, './processed_tensors/train/real')

    print("\n--- Extracting Training Data (FAKE) ---")
    for path in train_fake:
        extractor.process_video(path, './processed_tensors/train/fake')

    print("\n--- Extracting Validation Data (REAL) ---")
    for path in val_real:
        extractor.process_video(path, './processed_tensors/val/real')

    print("\n--- Extracting Validation Data (FAKE) ---")
    for path in val_fake:
        extractor.process_video(path, './processed_tensors/val/fake')

    print("\n[System] Phase 1: Data Extraction & Balancing Complete.")