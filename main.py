from ultralytics import YOLO
import cv2
import threading
import queue
import time
import numpy as np
from datetime import datetime
import winsound  # For buzzer sound on Windows

from config import CONFIG

merch_ids = CONFIG["merchandise_classes"]
person_ids = CONFIG["person_classes"]
CONFIG['stream_mode'] = False
CONFIG['video_source'] = "videos/vid4h.mp4"


class BathroomMonitor:
    """
    Multi-threaded bathroom entrance monitoring system with overlap-based zone 
    detection. Detects merchandise in bathroom zones and triggers immediate 
    audio alerts.

    Features:
    - Real-time YOLO object detection using YOLOv8n-OI7
    - Overlap-based zone monitoring (triggers on any bounding box intersection)
    - Immediate alerts for merchandise detection (regardless of person
      presence)
    - Audio alerts using winsound (announcement.wav file or buzzer fallback)
    - Multi-threaded architecture for smooth performance
    - UI scaling based on video resolution
    - Configurable statistics overlay with custom scaling
    - IP camera stream support with automatic reconnection
    - Stream error handling and recovery
    """

    def __init__(self, config=None):
        """Initialize with configuration dictionary"""
        # Use provided config or default CONFIG
        self.config = config if config is not None else CONFIG

        # Stream configuration
        self.stream_mode = self.config["stream_mode"]
        self.source = self.config["ip_camera_url"] if self.stream_mode else self.config["video_source"]
        self.cap = None
        self.stream_reconnect_attempts = 0
        self.max_reconnect_attempts = self.config["max_reconnect_attempts"]
        self.reconnect_delay = self.config["reconnect_delay"]

        # Initialize video capture with error handling
        self._initialize_video_capture()

        # Set camera properties for fisheye if needed (only for camera sources)
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        # YOLO model for object detection
        self.model = YOLO(self.config["model_path"])

        # Merchandise classes (bags, backpacks, handbags, suitcases, etc.)
        self.merchandise_classes = self.config["merchandise_classes"]
        self.person_classes = self.config["person_classes"]

        # Bathroom entrance zone
        self.bathroom_zone = self.config["bathroom_zone"]

        # Display toggles are now handled directly in annotation function

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
        self.show_stats = self.config["show_stats"]
        self.stats_scale_factor = self.config["stats_scale_factor"]

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

    def _initialize_video_capture(self):
        """Initialize video capture with stream mode support and error handling"""
        if self.stream_mode:
            print(f"🔄 Stream mode enabled - attempting to connect to: {self.source}")
            self._connect_to_stream()
        else:
            print(f"📹 Initializing video source: {self.source}")
            self.cap = cv2.VideoCapture(self.source)

            # Verify video source is accessible
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.source}")

            # Set camera properties for fisheye if needed (only for camera sources)
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            print("✅ Video source initialized successfully")

    def _connect_to_stream(self):
        """Connect to IP camera stream with retry logic"""
        while self.stream_reconnect_attempts < self.max_reconnect_attempts:
            try:
                print(f"🔄 Attempting to connect to stream (attempt {self.stream_reconnect_attempts + 1}/{self.max_reconnect_attempts})")

                # Release existing capture if any
                if self.cap is not None:
                    self.cap.release()

                # Try to connect to stream
                self.cap = cv2.VideoCapture(self.source)

                # Test if we can read a frame
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print("✅ Successfully connected to stream")
                        self.stream_reconnect_attempts = 0  # Reset counter on success
                        return
                    else:
                        print("⚠️  Stream opened but cannot read frames")
                else:
                    print("⚠️  Cannot open stream")

                self.stream_reconnect_attempts += 1

                if self.stream_reconnect_attempts < self.max_reconnect_attempts:
                    print(f"⏳ Waiting {self.reconnect_delay} seconds before retry...")
                    time.sleep(self.reconnect_delay)

            except Exception as e:
                print(f"❌ Stream connection error: {e}")
                self.stream_reconnect_attempts += 1

                if self.stream_reconnect_attempts < self.max_reconnect_attempts:
                    print(f"⏳ Waiting {self.reconnect_delay} seconds before retry...")
                    time.sleep(self.reconnect_delay)

        # If we get here, all attempts failed
        raise ConnectionError(f"Failed to connect to stream after {self.max_reconnect_attempts} attempts")

    def _check_stream_health(self):
        """Check if stream is still healthy and reconnect if needed"""
        if not self.stream_mode:
            return True

        if self.cap is None or not self.cap.isOpened():
            print("🔄 Stream disconnected, attempting to reconnect...")
            try:
                self._connect_to_stream()
                return True
            except ConnectionError:
                print("❌ Stream reconnection failed, shutting down...")
                return False

        return True

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
        """Thread function: Capture frames from camera/video/stream with error handling"""
        consecutive_failures = 0
        max_consecutive_failures = 10

        while self.running:
            # Check stream health for IP cameras
            if self.stream_mode and not self._check_stream_health():
                print("❌ Stream health check failed, stopping capture...")
                self.stop()
                break

            if not self.cap.isOpened():
                print("❌ Video capture not accessible")
                if self.stream_mode:
                    print("🔄 Attempting stream reconnection...")
                    if not self._check_stream_health():
                        self.stop()
                        break
                else:
                    break

            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"⚠️  Failed to read frame (attempt {consecutive_failures}/{max_consecutive_failures})")

                if self.stream_mode:
                    # For streams, try to reconnect
                    print("🔄 Stream read failed, attempting reconnection...")
                    if not self._check_stream_health():
                        print("❌ Stream reconnection failed, stopping...")
                        self.stop()
                        break
                    else:
                        consecutive_failures = 0  # Reset on successful reconnection
                        continue

                elif isinstance(self.source, str) and not self.stream_mode:
                    # If video file, restart from beginning
                    print("📹 End of video reached, restarting...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()

                    if ret:
                        consecutive_failures = 0  # Reset on success
                    else:
                        print("❌ Failed to restart video file")
                        if consecutive_failures >= max_consecutive_failures:
                            print("❌ Too many consecutive failures, stopping...")
                            self.stop()
                            break
                        continue
                else:
                    # For webcam or other sources
                    if consecutive_failures >= max_consecutive_failures:
                        print("❌ Too many consecutive failures, stopping...")
                        self.stop()
                        break
                    time.sleep(0.1)  # Brief pause before retry
                    continue
            else:
                consecutive_failures = 0  # Reset on successful frame read

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
                classes=self.merchandise_classes + self.person_classes,
                conf=self.config["confidence_threshold"],
                verbose=False,
                max_det=self.config["max_detections"]
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
        """Thread function: Monitor bathroom zone for merchandise using overlap detection.
        Triggers alarms whenever merchandise is detected in zone, regardless of person detection."""
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

            # Debug: Track detected classes
            detected_classes = set()

            for result in self.current_results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Debug: Track all detected classes
                    detected_classes.add(cls)

                    # Check if bounding box overlaps with bathroom zone
                    # Overlap occurs if boxes intersect (any part of detection box touches zone)
                    # Logic: box1 overlaps box2 if (x1 < x2_zone AND x2 > x1_zone AND y1 < y2_zone AND y2 > y1_zone)
                    in_zone = (x1 < zone_x2 and x2 > zone_x1 and
                              y1 < zone_y2 and y2 > zone_y1)

                    if in_zone:
                        if cls in self.person_classes:
                            people_in_zone.append((x1, y1, x2, y2, conf))
                            #print(f"DEBUG: Person detected overlapping zone - Class ID: {cls}, Conf: {conf:.2f}")
                        elif cls in self.merchandise_classes:
                            merchandise_in_zone.append((x1, y1, x2, y2, conf, cls))
                            #print(f"DEBUG: Merchandise detected overlapping zone - Class ID: {cls}, Conf: {conf:.2f}")

            # Debug: Print detected classes every few seconds
            if hasattr(self, '_last_debug_time'):
                if time.time() - self._last_debug_time > 5.0:  # Every 5 seconds
                    if detected_classes:
                        #print(f"DEBUG: All detected classes: {sorted(detected_classes)}")
                        #print(f"DEBUG: Looking for person classes: {self.person_classes}")
                        #print(f"DEBUG: Looking for merchandise classes (first 10): {self.merchandise_classes[:10]}...")
                        self._last_debug_time = time.time()
            else:
                self._last_debug_time = time.time()

            # Check for merchandise in zone and trigger alarm (regardless of person detection)
            if merchandise_in_zone and self._should_announce():
                self._make_announcement(len(merchandise_in_zone))
                self.stats['merchandise_detected'] += len(merchandise_in_zone)
                self.stats['announcements_made'] += 1
                # Count people if detected, otherwise count as 1 incident
                self.stats['customers_scanned'] += len(people_in_zone) if people_in_zone else 1

                # Enhanced debug output
                if people_in_zone:
                    print(f"🚨 ALARM: {len(merchandise_in_zone)} merchandise items detected with {len(people_in_zone)} people in zone")
                else:
                    print(f"🚨 ALARM: {len(merchandise_in_zone)} merchandise items detected in zone (no person currently detected)")

            time.sleep(1.0)  # Monitor every 1 second to allow audio to complete

    def _annotate_frame(self, frame, results):
        """Annotate frame with detection boxes and labels based on toggle settings"""
        annotated_frame = frame.copy()

        # Get annotation toggles
        annotations = self.config.get("annotations", {})
        show_persons = annotations.get("persons", True)
        show_items = annotations.get("items", True)

        # Note: Bathroom zone is now drawn by _draw_bathroom_zone method

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

                # Determine if we should show this detection based on toggles
                should_show = False

                if cls in self.person_classes:
                    # Show person boxes if persons toggle is enabled
                    should_show = show_persons

                elif cls in self.merchandise_classes:
                    # Show item boxes if items toggle is enabled
                    should_show = show_items

                else:
                    # Other classes - show if items toggle is enabled
                    should_show = show_items

                if not should_show:
                    continue

                # Choose color and label based on class
                if cls in self.person_classes:
                    color = (0, 255, 0)  # Green for person
                    label = f"Person {conf:.2f}"
                elif cls in self.merchandise_classes:
                    color = (0, 0, 255)  # Red for merchandise
                    label = f"Item {conf:.2f}"
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
        """Draw bathroom monitoring zone on frame if enabled"""
        # Check if bathroom zone annotation is enabled
        annotations = self.config.get("annotations", {})
        show_bathroom_zone = annotations.get("bathroom_zone", True)

        if not show_bathroom_zone:
            return  # Don't draw zone if disabled

        h, w = frame.shape[:2]

        # Convert relative coordinates to absolute
        x1 = int(self.bathroom_zone['x1'] * w)
        y1 = int(self.bathroom_zone['y1'] * h)
        x2 = int(self.bathroom_zone['x2'] * w)
        y2 = int(self.bathroom_zone['y2'] * h)

        # Draw zone rectangle with scaled thickness
        zone_thickness = max(2, int(3 * self.scale_factor))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), zone_thickness)

        # Zone label is commented out to reduce visual clutter
        # Uncomment below lines to show "BATHROOM ZONE" text
        # zone_text_scale = max(0.4, 0.7 * self.scale_factor)
        # zone_text_thickness = max(1, int(2 * self.scale_factor))
        # label_offset = max(5, int(10 * self.scale_factor))
        # cv2.putText(frame, "BATHROOM ZONE", (x1, y1 - label_offset),
        #            cv2.FONT_HERSHEY_SIMPLEX, zone_text_scale, (255, 255, 0), zone_text_thickness)

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

                # Calculate center point (for tracking purposes)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check if bounding box overlaps with bathroom zone
                # Any part of detection box touching zone triggers detection
                in_zone = (x1 < zone_x2 and x2 > zone_x1 and
                          y1 < zone_y2 and y2 > zone_y1)

                if cls in self.person_classes and in_zone:
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
                if distance < 250:
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
                winsound.PlaySound("audio/speech1.wav", winsound.SND_FILENAME | winsound.SND_NOSTOP)

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

    print("🚀 Initializing Bathroom Monitor...")
    print("📋 Configuration:")
    print(f"   Model: {CONFIG['model_path']}")
    print(f"   Stream Mode: {'✅ Enabled' if CONFIG['stream_mode'] else '❌ Disabled'}")

    if CONFIG['stream_mode']:
        print(f"   Source: {CONFIG['ip_camera_url']} (IP Camera)")
    else:
        print(f"   Source: {CONFIG['video_source']} (Local)")

    zone = CONFIG['bathroom_zone']
    print(f"   Zone: ({zone['x1']:.2f}, {zone['y1']:.2f}) to ({zone['x2']:.2f}, {zone['y2']:.2f})")
    print(f"   Show Stats: {'✅ Yes' if CONFIG['show_stats'] else '❌ No'}")

    # Display annotation toggles
    annotations = CONFIG.get('annotations', {})
    print(f"   Annotation Toggles:")
    print(f"     Bathroom Zone: {'✅ Visible' if annotations.get('bathroom_zone', True) else '❌ Hidden'}")
    print(f"     Person Boxes: {'✅ Visible' if annotations.get('persons', True) else '❌ Hidden'}")
    print(f"     Item Boxes: {'✅ Visible' if annotations.get('items', True) else '❌ Hidden'}")



    # Create and start monitor
    try:
        monitor = BathroomMonitor(CONFIG)
    except (ValueError, ConnectionError) as e:
        print(f"❌ Failed to initialize monitor: {e}")
        return

    try:
        print("🎯 Starting monitoring system...")
        monitor.start()

        print("✅ Monitoring system started successfully!")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to show statistics")
        print("   - Close video window to stop")

        if CONFIG['stream_mode']:
            print("🌐 Stream monitoring active - system will auto-reconnect if stream drops")

        # Keep main thread alive
        while monitor.running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⚠️  Keyboard interrupt received...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        print("🛑 Stopping monitoring system...")
        monitor.stop()
        print("✅ System stopped successfully")


if __name__ == "__main__":
    main()