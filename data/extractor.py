import cv2
import torch
import os
import glob
import random
from facenet_pytorch import MTCNN
from torchvision import transforms


class DeepfakeDataExtractor:
    def __init__(self, device='cuda', image_size=224, seq_length=16):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.seq_length = seq_length

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
        # FIX 1: Chunk frames to prevent CUDA Illegal Memory Access
        batch_size = 32
        for i in range(0, len(frames), batch_size):
            chunk = frames[i:i + batch_size]
            try:
                faces = self.mtcnn(chunk)
                if faces is not None:
                    for face in faces:
                        if face is not None:
                            valid_faces.append(self.transform(face / 255.0))
            except Exception as e:
                # If a corrupted frame breaks MTCNN, just skip it and keep going
                continue

        if len(valid_faces) < self.seq_length: return

        face_tensor = torch.stack(valid_faces)
        num_sequences = len(face_tensor) // self.seq_length

        # --- THE SAFETY CAP: MAX 4 SEQUENCES PER VIDEO ---
        max_sequences = 2
        num_sequences = min(num_sequences, max_sequences)
        # -------------------------------------------------

        sequences = face_tensor[:num_sequences * self.seq_length].view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )

        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"Processed {video_name} -> {num_sequences} seqs saved to {output_dir.split('/')[-2]}")


if __name__ == "__main__":
    extractor = DeepfakeDataExtractor(device='cuda', seq_length=16)
    source_base_dir = "/workspace/ff-c23/FaceForensics++_C23"
    output_base_dir = "./processed_tensors"

    # THE STORAGE SAVER: Only process Originals, Deepfakes, and NeuralTextures
    folder_mapping = {
        'original': 'real',
        'Deepfakes': 'fake',
        'NeuralTextures': 'fake'
        # 'Face2Face': 'fake',
        # 'FaceSwap': 'fake',
        # 'FaceShifter': 'fake',
        # 'DeepFakeDetection': 'fake'
    }

    for input_folder, label in folder_mapping.items():
        source_dir = os.path.join(source_base_dir, input_folder)
        if os.path.exists(source_dir):
            video_files = glob.glob(os.path.join(source_dir, "**/*.mp4"), recursive=True)
            print(f"Found {len(video_files)} videos in {input_folder}...")

            for video_path in video_files:
                # FIX 2: Dynamically split into 80% train and 20% val folders
                split_folder = "val" if random.random() < 0.2 else "train"
                output_dir = os.path.join(output_base_dir, split_folder, label)
                extractor.process_video(video_path, output_dir)