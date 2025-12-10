"""
YOLOv8 Training Pipeline for Defect Detection
Loads a pre-trained YOLOv8 model from COCO dataset and fine-tunes it on custom data.
"""

from ultralytics import YOLO


def train():
    # Load YOLOv8 pre-trained model (trained on COCO dataset)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), 
    #          yolov8l.pt (large), yolov8x.pt (extra-large)
    model = YOLO("yolov8n.pt")  # Using nano model for faster training, change as needed

    # Train the model on custom dataset
    results = model.train(
        data="yolov8.yaml",           # Dataset configuration file
        epochs=100,                    # Number of training epochs
        imgsz=640,                     # Input image size
        batch=16,                      # Batch size (adjust based on GPU memory)
        project="runs/detect",         # Project directory
        name="train",                  # Run name
        exist_ok=True,                 # Overwrite existing run
        pretrained=True,               # Use pre-trained weights
        optimizer="auto",              # Optimizer (auto selects best)
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate factor
        patience=50,                   # Early stopping patience
        save=True,                     # Save checkpoints
        save_period=-1,                # Save checkpoint every x epochs (-1 = disabled)
        device=0,                      # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=8,                     # Number of data loader workers
        verbose=True,                  # Verbose output
    )

    print("\nTraining complete!")
    print(f"Best weights saved to: runs/detect/train/weights/best.pt")
    print(f"Last weights saved to: runs/detect/train/weights/last.pt")

    return results


if __name__ == "__main__":
    train()

