import cv2
import torch
import os
import glob
from facenet_pytorch import MTCNN
from torchvision import transforms


class DeepfakeDataExtractor:
    def __init__(self, device='cuda', image_size=224, seq_length=16):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        self.seq_length = seq_length

        # Initialize MTCNN on the GPU for fast batched processing
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=40,
            keep_all=False,
            post_process=False,
            device=self.device
        )

        # Standard normalization for pre-trained ViTs or ResNets
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

        # Skip if already processed (resumes gracefully if RunPod preempts)
        if os.path.exists(output_path):
            return

        frames = self.extract_frames(video_path)
        if not frames: return

        # Batch process frames through MTCNN
        # Note: If VRAM OOMs on 48GB, chunk 'frames' list into blocks of 300
        faces = self.mtcnn(frames)

        valid_faces = []
        for face in faces:
            if face is not None:
                face_normalized = self.transform(face / 255.0)
                valid_faces.append(face_normalized)

        if len(valid_faces) < self.seq_length:
            return

        # Stack into a single tensor: [Total_Faces, 3, 224, 224]
        face_tensor = torch.stack(valid_faces)

        # Chunk into sequences: [Num_Sequences, seq_length, 3, 224, 224]
        num_sequences = len(face_tensor) // self.seq_length
        sequences = face_tensor[:num_sequences * self.seq_length].view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )

        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, output_path)
        print(f"Processed {video_name} -> {num_sequences} sequences extracted.")


# --- Bulk Execution Example ---
if __name__ == "__main__":
    extractor = DeepfakeDataExtractor(device='cuda', seq_length=16)

    # Example directory structure:
    # /datasets/ILLUSION/fake/video1.mp4
    # /datasets/ILLUSION/real/video2.mp4

    # 1. SET THIS PATH: Where you will unzip the FF++ Kaggle download
    source_base_dir = "/workspace/FaceForensics_Dataset"

    # 2. SET THIS PATH: Where the .pt tensors will be saved
    output_base_dir = "/workspace/defensive-kyc/processed_tensors"

    for category in ['real', 'fake']:
        source_dir = os.path.join(source_base_dir, category)
        output_dir = os.path.join(output_base_dir, category)

        if os.path.exists(source_dir):
            video_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            print(f"Found {len(video_files)} videos in {category} folder.")

            for video_path in video_files:
                extractor.process_video(video_path, output_dir)