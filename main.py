from ultralytics import YOLO
import cv2
import threading
import queue
import time
import numpy as np
from datetime import datetime
import winsound  # For buzzer sound on Windows

# Remove DeepSORT - using simple detection only

# Audio announcements using winsound


class BathroomMonitor:
    """
    Multi-threaded bathroom entrance monitoring system with fisheye camera support.
    Detects customers approaching bathrooms, scans for merchandise, and tracks abandoned items.

    Features:
    - Real-time YOLO object detection
    - Configurable bathroom zone monitoring
    - Audio alerts using winsound (announcement.wav file or buzzer fallback)
    - Multi-threaded architecture for smooth performance
    - UI scaling based on video resolution
    - Configurable statistics overlay with custom scaling
    """

    def __init__(self, model_path: str, source=0, bathroom_zone=None, show_stats=True, stats_scale_factor=1.0):
        # Video capture setup
        self.source = source
        self.cap = cv2.VideoCapture(source)

        # Verify video source is accessible
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        # Set camera properties for fisheye if needed (only for camera sources)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        # YOLO model for object detection
        self.model = YOLO(model_path)

        # Removed DeepSORT tracker - using simple detection only

        # Merchandise classes (bags, backpacks, handbags, suitcases, etc.)
        self.merchandise_classes = [24,25,26,27,28,29,39,49,63,65,67,76]  # Common merchandise items
        self.person_class = 0  # Person class in COCO dataset

        # Bathroom entrance zone (default to center area)
        if bathroom_zone is None:
            self.bathroom_zone = {
                'x1': 0.3, 'y1': 0.2,  # Relative coordinates (0-1)
                'x2': 0.7, 'y2': 0.8
            }
        else:
            self.bathroom_zone = bathroom_zone

        # Threading components
        self.frame_queue = queue.Queue(maxsize=20)
        self.detection_queue = queue.Queue(maxsize=20)
        self.display_queue = queue.Queue(maxsize=20)

        self.capture_thread = None
        self.detection_thread = None
        self.display_thread = None
        self.monitoring_thread = None

        # Control flags
        self.running = False
        self.stopped = False

        # Tracking and monitoring state
        self.current_frame = None
        self.current_results = None
        self.annotated_frame = None

        # Audio announcement management
        self.is_speaking = False
        self.last_announcement_time = 0
        self.min_announcement_interval = 2.0  # Minimum 2 seconds between announcements

        # Stats UI configuration
        self.show_stats = show_stats
        self.stats_scale_factor = stats_scale_factor

        # UI scaling factors (will be set based on video resolution)
        self.scale_factor = 0.5
        self.text_scale = 0.5
        self.thickness = 0.1
        self.font_thickness = 1

        # Statistics
        self.stats = {
            'customers_scanned': 0,
            'merchandise_detected': 0,
            'announcements_made': 0,
            'abandoned_items': 0,
            'theft_deterred': 0
        }

    def start(self):
        """Start all monitoring threads"""
        if self.running:
            return

        self.running = True
        self.stopped = False

        # Calculate UI scaling factors based on video resolution
        self._calculate_scaling_factors()

        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.detection_thread = threading.Thread(target=self._detect_objects, daemon=True)
        self.display_thread = threading.Thread(target=self._display_frames, daemon=True)
        self.monitoring_thread = threading.Thread(target=self._monitor_bathroom_zone, daemon=True)

        self.capture_thread.start()
        self.detection_thread.start()
        self.display_thread.start()
        self.monitoring_thread.start()

        print("Bathroom monitoring system started...")
        print("Press 'q' to quit, 's' to show stats")

    def _calculate_scaling_factors(self):
        """Calculate UI scaling factors based on video resolution"""
        # Get a sample frame to determine resolution
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]

            # Base resolution for scaling (1920x1080)
            base_width = 1920
            base_height = 1080

            # Calculate scale factor based on width (primary factor)
            width_scale = w / base_width
            height_scale = h / base_height

            # Use the smaller scale to ensure UI elements fit
            self.scale_factor = min(width_scale, height_scale)

            # Ensure minimum scale factor
            self.scale_factor = max(0.3, self.scale_factor)

            # Calculate derived scaling factors
            self.text_scale = max(0.3, 0.5 * self.scale_factor)
            self.thickness = max(1, int(2 * self.scale_factor))
            self.font_thickness = max(1, int(2 * self.scale_factor))

            print(f"Video resolution: {w}x{h}")
            print(f"Scale factor: {self.scale_factor:.2f}")
            print(f"Text scale: {self.text_scale:.2f}")
            print(f"Line thickness: {self.thickness}")

            # Reset video position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Warning: Could not read frame for scaling calculation, using defaults")

    def stop(self):
        """Stop all threads and cleanup"""
        self.running = False
        self.stopped = True

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        print("\nBathroom monitoring system stopped.")
        self._print_stats()

    def _capture_frames(self):
        """Thread function: Capture frames from fisheye camera"""
        while self.running:
            if not self.cap.isOpened():
                print("Error: Camera not accessible")
                break

            ret, frame = self.cap.read()
            if not ret:
                # If video file, restart from beginning
                if isinstance(self.source, str):
                    print("End of video reached, restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                if not ret:
                    print("Error: Failed to read frame")
                    break

            # Store current frame for other threads
            self.current_frame = frame.copy()

            # Add frame to detection queue (non-blocking)
            try:
                self.frame_queue.put(frame, timeout=0.01)
            except queue.Full:
                pass  # Skip frame if queue is full

            time.sleep(0.01)  # Small delay to prevent overwhelming

    def _detect_objects(self):
        """Thread function: Run YOLO detection on frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Run YOLO detection
            results = self.model.predict(
                frame,
                classes=self.merchandise_classes + [self.person_class],
                conf=0.5,
                verbose=False
            )

            # Store current results for monitoring thread
            self.current_results = results

            # Create annotated frame (no tracking)
            annotated_frame = self._annotate_frame(frame, results)

            # Add to display queue
            try:
                self.display_queue.put((annotated_frame, results), timeout=0.01)
            except queue.Full:
                pass  # Skip if display queue is full

    def _display_frames(self):
        """Thread function: Display annotated frames in OpenCV window"""
        cv2.namedWindow('Bathroom Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Bathroom Monitor', 1280, 720)

        while self.running:
            try:
                annotated_frame, _ = self.display_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Store for other threads
            self.annotated_frame = annotated_frame

            # Add bathroom zone overlay
            self._draw_bathroom_zone(annotated_frame)

            # Add statistics overlay
            self._draw_stats_overlay(annotated_frame)

            # Display frame
            cv2.imshow('Bathroom Monitor', annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            elif key == ord('s'):
                self._print_stats()

    def _monitor_bathroom_zone(self):
        """Thread function: Monitor bathroom zone for people with merchandise using simple detection"""
        while self.running:
            if self.current_frame is None or self.current_results is None:
                time.sleep(0.1)
                continue

            # Get frame dimensions
            h, w = self.current_frame.shape[:2]

            # Convert bathroom zone to absolute coordinates
            zone_x1 = int(self.bathroom_zone['x1'] * w)
            zone_y1 = int(self.bathroom_zone['y1'] * h)
            zone_x2 = int(self.bathroom_zone['x2'] * w)
            zone_y2 = int(self.bathroom_zone['y2'] * h)

            # Check current detections in bathroom zone
            people_in_zone = []
            merchandise_in_zone = []

            for result in self.current_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Check if in bathroom zone
                    in_zone = (zone_x1 <= center_x <= zone_x2 and
                              zone_y1 <= center_y <= zone_y2)

                    if in_zone:
                        if cls == self.person_class:
                            people_in_zone.append((x1, y1, x2, y2, conf))
                        elif cls in self.merchandise_classes:
                            merchandise_in_zone.append((x1, y1, x2, y2, conf, cls))

            # Check for people with merchandise in zone and trigger alarm
            if people_in_zone and merchandise_in_zone and self._should_announce():
                self._make_announcement(len(merchandise_in_zone))
                self.stats['merchandise_detected'] += len(merchandise_in_zone)
                self.stats['announcements_made'] += 1
                self.stats['customers_scanned'] += len(people_in_zone)

            time.sleep(1.0)  # Monitor every 1 second to allow audio to complete

    def _annotate_frame(self, frame, results):
        """Annotate frame with detection boxes and labels"""
        annotated_frame = frame.copy()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Choose color based on class
                if cls == self.person_class:
                    color = (0, 255, 0)  # Green for person
                    label = f"Person {conf:.2f}"
                elif cls in self.merchandise_classes:
                    color = (0, 0, 255)  # Red for merchandise
                    class_names = {24: "Backpack", 25: "Umbrella", 26: "Handbag",
                                 27: "Tie", 28: "Suitcase", 29: "Frisbee"}
                    label = f"{class_names.get(cls, 'Item')} {conf:.2f}"
                else:
                    color = (255, 0, 0)  # Blue for other
                    label = f"Class {cls} {conf:.2f}"

                # Draw bounding box with scaled thickness
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.thickness)

                # Draw label with scaled text
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.font_thickness)[0]
                label_padding = max(5, int(10 * self.scale_factor))
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - label_padding),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - max(3, int(5 * self.scale_factor))),
                          cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (255, 255, 255), self.font_thickness)

        return annotated_frame

    # Removed tracking annotation method - using simple detection only

    # Removed tracking helper methods - using simple detection only

    def _draw_bathroom_zone(self, frame):
        """Draw bathroom monitoring zone on frame"""
        h, w = frame.shape[:2]

        # Convert relative coordinates to absolute
        x1 = int(self.bathroom_zone['x1'] * w)
        y1 = int(self.bathroom_zone['y1'] * h)
        x2 = int(self.bathroom_zone['x2'] * w)
        y2 = int(self.bathroom_zone['y2'] * h)

        # Draw zone rectangle with scaled thickness
        zone_thickness = max(2, int(3 * self.scale_factor))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), zone_thickness)

        # Add zone label with scaled text
        zone_text_scale = max(0.4, 0.7 * self.scale_factor)
        zone_text_thickness = max(1, int(2 * self.scale_factor))
        label_offset = max(5, int(10 * self.scale_factor))
        cv2.putText(frame, "BATHROOM ZONE", (x1, y1 - label_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, zone_text_scale, (255, 255, 0), zone_text_thickness)

    def _draw_stats_overlay(self, frame):
        """Draw statistics overlay on frame"""
        # Check if stats display is enabled
        if not self.show_stats:
            return

        # Use only user-specified scale factor (independent of video resolution)
        user_scale = self.stats_scale_factor

        # Base dimensions (designed for normal scale = 1.0)
        base_box_margin = 10
        base_box_width = 400
        base_box_height = 200
        base_text_scale = 0.6
        base_text_thickness = 2
        base_line_spacing = 25
        base_text_margin_x = 10
        base_text_margin_y = 30

        # Calculate scaled dimensions using only user scale factor
        box_margin = max(2, int(base_box_margin * user_scale))
        box_width = max(50, int(base_box_width * user_scale))
        box_height = max(30, int(base_box_height * user_scale))

        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_margin, box_margin),
                     (box_margin + box_width, box_margin + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Add statistics text with user-scaled font
        stats_text = [
            f"Customers Scanned: {self.stats['customers_scanned']}",
            f"Merchandise Detected: {self.stats['merchandise_detected']}",
            f"Announcements Made: {self.stats['announcements_made']}",
            f"Abandoned Items: {self.stats['abandoned_items']}",
            f"Theft Deterred: {self.stats['theft_deterred']}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]

        stats_text_scale = max(0.1, base_text_scale * user_scale)
        stats_text_thickness = max(1, int(base_text_thickness * user_scale))
        line_spacing = max(5, int(base_line_spacing * user_scale))
        text_start_x = box_margin + max(1, int(base_text_margin_x * user_scale))
        text_start_y = box_margin + max(5, int(base_text_margin_y * user_scale))

        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (text_start_x, text_start_y + i * line_spacing),
                       cv2.FONT_HERSHEY_SIMPLEX, stats_text_scale, (255, 255, 255), stats_text_thickness)

    def _process_detections(self, results):
        """Process YOLO detections for bathroom monitoring"""
        current_time = time.time()
        h, w = self.current_frame.shape[:2] if self.current_frame is not None else (720, 1280)

        # Convert bathroom zone to absolute coordinates
        zone_x1 = int(self.bathroom_zone['x1'] * w)
        zone_y1 = int(self.bathroom_zone['y1'] * h)
        zone_x2 = int(self.bathroom_zone['x2'] * w)
        zone_y2 = int(self.bathroom_zone['y2'] * h)

        people_detected = []
        merchandise_detected = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check if detection is in bathroom zone
                in_zone = (zone_x1 <= center_x <= zone_x2 and
                          zone_y1 <= center_y <= zone_y2)

                if cls == self.person_class and in_zone:
                    people_detected.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'conf': conf,
                        'time': current_time
                    })

                elif cls in self.merchandise_classes:
                    merchandise_detected.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'conf': conf,
                        'class': cls,
                        'time': current_time,
                        'in_zone': in_zone
                    })

        # Process people entering bathroom zone
        for person in people_detected:
            self.stats['customers_scanned'] += 1

            # Check for merchandise near this person
            person_merchandise = []
            for item in merchandise_detected:
                # Calculate distance between person and merchandise
                person_center = person['center']
                item_center = item['center']
                distance = np.sqrt((person_center[0] - item_center[0])**2 +
                                 (person_center[1] - item_center[1])**2)

                # If merchandise is close to person (within 100 pixels)
                if distance < 100:
                    person_merchandise.append(item)

            # If person has merchandise, make announcement
            if person_merchandise and self._should_announce():
                self._make_announcement(len(person_merchandise))
                self.stats['merchandise_detected'] += len(person_merchandise)
                self.stats['announcements_made'] += 1

                # Track merchandise with this person
                person_id = f"person_{current_time}"
                self.merchandise_detections[person_id] = {
                    'items': person_merchandise,
                    'time': current_time,
                    'announced': True
                }

        # Track standalone merchandise (potentially abandoned)
        for item in merchandise_detected:
            if not item['in_zone']:
                continue

            item_id = f"item_{item['center'][0]}_{item['center'][1]}"

            # Check if this is a new abandoned item
            if item_id not in self.abandoned_items:
                self.abandoned_items[item_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'bbox': item['bbox'],
                    'class': item['class'],
                    'stable_count': 1
                }
            else:
                # Update existing item
                self.abandoned_items[item_id]['last_seen'] = current_time
                self.abandoned_items[item_id]['stable_count'] += 1

    def _check_abandoned_items(self):
        """Check for items that have been abandoned for too long"""
        current_time = time.time()
        abandoned_threshold = 30  # 30 seconds

        items_to_remove = []

        for item_id, item_data in self.abandoned_items.items():
            time_abandoned = current_time - item_data['first_seen']
            time_since_seen = current_time - item_data['last_seen']

            # If item has been stable for threshold time, consider it abandoned
            if (time_abandoned > abandoned_threshold and
                item_data['stable_count'] > 10 and
                time_since_seen < 5):  # Still being detected

                if not item_data.get('reported', False):
                    self.stats['abandoned_items'] += 1
                    self.stats['theft_deterred'] += 1
                    self.abandoned_items[item_id]['reported'] = True

                    print(f"ALERT: Abandoned merchandise detected for {time_abandoned:.1f}s")

            # Remove items not seen for a while
            elif time_since_seen > 10:
                items_to_remove.append(item_id)

        # Clean up old items
        for item_id in items_to_remove:
            del self.abandoned_items[item_id]

    def _should_announce(self):
        """Check if we should make an announcement (not currently speaking and enough time passed)"""
        current_time = time.time()
        return (not self.is_speaking and
                current_time - self.last_announcement_time >= self.min_announcement_interval)

    def _make_announcement(self, item_count):
        """Make audio announcement about merchandise detection"""
        # Update timing
        self.last_announcement_time = time.time()
        self.is_speaking = True

        if item_count == 1:
            message = "Attention: Merchandise is not permitted in the bathroom. Please leave your item outside."
        else:
            message = f"Attention: {item_count} items detected. Merchandise is not permitted in the bathroom. Please leave your items outside."

        print(f"ANNOUNCEMENT: {message}")

        # Play audio file
        def play_audio_file():
            try:
                # Play announcement.wav file with SND_NOSTOP to prevent interruption
                winsound.PlaySound("speech1.wav", winsound.SND_FILENAME | winsound.SND_NOSTOP)

                # Mark speaking as complete
                self.is_speaking = False

            except Exception as e:
                print(f"Audio File Error: {e}")
                # Fallback to buzzer beeps if audio file fails
                try:
                    for i in range(3):
                        winsound.Beep(1000, 200)  # 1000Hz for 200ms
                        if i < 2:
                            time.sleep(0.1)
                except:
                    pass
                self.is_speaking = False

        threading.Thread(target=play_audio_file, daemon=True).start()

    def _print_stats(self):
        """Print current statistics"""
        print("\n" + "="*50)
        print("BATHROOM MONITORING STATISTICS")
        print("="*50)
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("="*50)


def main():
    """Main function to run the bathroom monitoring system"""
    # Configuration
    model_path = "yolo11n.pt"  # Path to YOLO model (will download if not exists)
    video_source = "videos/vid5.mp4"  # Use 0 for webcam, or path to video file

    # Custom bathroom zone (optional)
    bathroom_zone = {
        'x1': 0.01, 'y1': 0.01,  # Top-left corner (relative coordinates)
        'x2': 0.99, 'y2': 0.6   # Bottom-right corner (relative coordinates)
    }

    # Stats UI Configuration
    show_stats = False  # True: Show statistics overlay, False: Hide statistics
    stats_scale_factor = 0.2  # Scale factor for stats UI (1.0 = normal, 1.5 = 50% larger, 0.8 = 20% smaller)

    # Create and start monitor
    monitor = BathroomMonitor(model_path, video_source, bathroom_zone, show_stats, stats_scale_factor)

    try:
        monitor.start()

        # Keep main thread alive
        while monitor.running:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()