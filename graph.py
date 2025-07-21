from ultralytics import YOLO
import cv2
import threading
import queue
import time
import numpy as np
import torch

class OptimizedYOLODetector:
    def __init__(self):
        # Check GPU availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = torch.cuda.is_available()  # Half precision only on GPU
        
        print(f"Using device: {self.device}")
        if self.device == 'cpu':
            print("⚠️  Warning: No GPU detected. Performance will be slower.")
            print("   For better performance, install CUDA-enabled PyTorch.")
        else:
            print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        
        # Use YOLOv8x (extra-large) for highest accuracy
        self.model = YOLO('yolov8x.pt')
        
        # Configure model for speed
        self.model.overrides['verbose'] = False
        
        # Threading components (larger queues for YOLOv8x)
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=3)
        self.running = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Initialize webcam with optimized settings
        self.setup_camera()
        
    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        
        # Adjust resolution based on device capability
        if self.device == 'cuda':
            # Higher resolution for GPU
            width, height = 1280, 720
        else:
            # Lower resolution for CPU to maintain performance
            width, height = 640, 480
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Additional optimizations
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        # Verify actual settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}FPS")
        
    def capture_frames(self):
        """Separate thread for frame capture"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Skip frames if queue is full (maintain real-time performance)
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
    def process_detections(self):
        """Separate thread for YOLO inference"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                
                # Adaptive settings based on device
                if self.device == 'cuda':
                    # GPU settings - higher quality
                    imgsz = 640
                    conf = 0.3
                    max_det = 100
                else:
                    # CPU settings - optimized for speed
                    imgsz = 416
                    conf = 0.5
                    max_det = 50
                
                # YOLO inference with automatic device detection
                results = self.model(
                    frame,
                    imgsz=imgsz,
                    conf=conf,
                    iou=0.45,
                    max_det=max_det,
                    device=self.device,  # Automatically uses CPU or GPU
                    half=self.use_half,  # Half precision only if GPU available
                    verbose=False,
                    agnostic_nms=True
                )
                
                # Only add to result queue if not full
                if not self.result_queue.full():
                    self.result_queue.put((frame, results))
                    
            except queue.Empty:
                continue
                
    def calculate_fps(self):
        """Calculate and display FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update FPS every 30 frames
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            
    def draw_fps(self, frame):
        """Draw FPS counter and device info on frame"""
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Device: {self.device.upper()}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame
        
    def run(self):
        """Main detection loop"""
        # Start worker threads
        capture_thread = threading.Thread(target=self.capture_frames)
        detection_thread = threading.Thread(target=self.process_detections)
        
        capture_thread.daemon = True
        detection_thread.daemon = True
        
        capture_thread.start()
        detection_thread.start()
        
        # Set up display window
        cv2.namedWindow('YOLOv8x High Performance Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLOv8x High Performance Detection', 1280, 720)
        
        print("Starting YOLOv8x detection (highest accuracy)...")
        print("Press 'q' to quit, 'f' to toggle fullscreen")
        print("Note: GPU recommended for optimal performance")
        
        fullscreen = False
        last_frame = None
        
        while True:
            try:
                # Get latest detection result
                frame, results = self.result_queue.get(timeout=0.03)
                
                # Render detections
                rendered_frame = results[0].plot()
                
                # Add FPS counter
                rendered_frame = self.draw_fps(rendered_frame)
                
                last_frame = rendered_frame
                self.calculate_fps()
                
            except queue.Empty:
                # If no new results, show last frame to maintain smooth display
                if last_frame is not None:
                    rendered_frame = self.draw_fps(last_frame.copy())
                else:
                    continue
            
            # Display frame
            cv2.imshow('YOLOv8x High Performance Detection', rendered_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty('YOLOv8x High Performance Detection', 
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('YOLOv8x High Performance Detection', 
                                        cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        # Cleanup
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

# Alternative simple optimized version (if threading seems complex)
def simple_optimized_detection():
    """Simplified version optimized for YOLOv8x with automatic device detection"""
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_half = torch.cuda.is_available()
    
    print(f"Simple mode using device: {device}")
    
    # Use highest performing model
    model = YOLO('yolov8x.pt')
    
    # Initialize webcam with adaptive resolution
    cap = cv2.VideoCapture(0)
    if device == 'cuda':
        width, height = 1280, 720
    else:
        width, height = 640, 480
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    cv2.namedWindow('YOLOv8x High Performance', cv2.WINDOW_NORMAL)
    
    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    print("Using YOLOv8x - highest accuracy model")
    
    # Frame processing settings based on device
    frame_skip = 1 if device == 'cuda' else 2  # Skip frames on CPU for speed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        fps_counter += 1
        
        # Skip frames on CPU for better performance
        if device == 'cpu' and fps_counter % frame_skip != 0:
            continue
            
        # Adaptive inference settings
        if device == 'cuda':
            imgsz, conf = 640, 0.25
        else:
            imgsz, conf = 416, 0.5
            
        # High-performance inference settings
        results = model(
            frame, 
            imgsz=imgsz, 
            conf=conf,
            iou=0.45,
            device=device,  # Automatically detected
            half=use_half,  # Half precision only on GPU
            verbose=False
        )
        
        rendered_frame = results[0].plot()
        
        # Show FPS and device info
        current_time = time.time()
        fps = fps_counter / (current_time - start_time)
        cv2.putText(rendered_frame, f'YOLOv8x FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(rendered_frame, f'Device: {device.upper()}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('YOLOv8x High Performance', rendered_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("YOLOv8x High Performance Detection")
    print("=" * 50)
    
    # Check system capabilities
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print("   Optimal performance expected")
    else:
        print("⚠️  GPU Not Available - Running on CPU")
        print("   Performance will be slower but still functional")
    
    print("\nChoose detection mode:")
    print("1. Advanced threaded detection (recommended)")
    print("2. Simple high-performance detection")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        simple_optimized_detection()
    else:
        detector = OptimizedYOLODetector()
        detector.run()