import cv2
import torch
import os
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
        """Reads video and extracts all frames as a list of RGB numpy arrays."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV reads in BGR, we need RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames

    def process_video(self, video_path, output_dir):
        """Processes a single video into sequential tensor chunks."""
        video_name = os.path.basename(video_path).split('.')[0]
        frames = self.extract_frames(video_path)

        if not frames:
            print(f"Failed to read {video_path}")
            return

        print(f"Processing {len(frames)} frames from {video_name}...")

        # Batch process frames through MTCNN to maximize GPU usage
        # Note: For very long videos, you might need to chunk this to avoid VRAM OOM
        faces = self.mtcnn(frames)

        valid_faces = []
        for face in faces:
            if face is not None:
                # face is a tensor of shape [3, 224, 224], values 0-255
                face_normalized = self.transform(face / 255.0)
                valid_faces.append(face_normalized)

        if len(valid_faces) < self.seq_length:
            print(f"Not enough valid faces found in {video_name}. Skipping.")
            return

        # Stack into a single tensor: [Total_Faces, 3, 22 4, 224]
        face_tensor = torch.stack(valid_faces)

        # Chunk into sequences: [Num_Sequences, seq_length, 3, 224, 224]
        num_sequences = len(face_tensor) // self.seq_length
        sequences = face_tensor[:num_sequences * self.seq_length].view(
            num_sequences, self.seq_length, 3, self.image_size, self.image_size
        )

        # Save the sequences to disk
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}.pt")
        torch.save(sequences, output_path)
        print(f"Saved {num_sequences} sequences to {output_path}")


# --- Execution Example ---
if __name__ == "__main__":
    # Point this to your ILLUSION dataset directory
    input_video = "sample_deepfake.mp4"
    output_directory = "./processed_tensors/fake/"

    extractor = DeepfakeDataExtractor(device='cuda', seq_length=16)
    extractor.process_video(input_video, output_directory)