import os
import threading
import queue
import time
from datetime import datetime
import winsound
import logging
from pathlib import Path
try:
    import msvcrt  # For non-blocking key reads on Windows
except Exception:
    msvcrt = None

import torch

from .config import CONFIG

from ultralytics import YOLO
from .benchmark import Benchmark
import cv2
import numpy as np


class BathroomMonitor:
    """
    Multi-threaded bathroom entrance monitoring system with overlap-based zone 
    detection. Detects merchandise in bathroom zones and triggers immediate 
    audio alerts.

    Features:
    - Real-time YOLO object detection using YOLO11n
    - Overlap-based zone monitoring (triggers on any bounding box intersection)
    - Immediate alerts for merchandise detection (regardless of person
      presence)
    - Audio alerts using winsound
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
        #Model warmup
        print('Warming up YOLO...')
        img = torch.zeros(1,3,640,640)
        for i in range(5):
            self.model.predict(img, verbose=False)

        # Merchandise classes (bags, backpacks, handbags, suitcases, etc.)
        self.merchandise_classes = self.config["merchandise_classes"]
        self.person_classes = self.config["person_classes"]

        # Bathroom entrance zone
        self.bathroom_zone = self.config["bathroom_zone"]
        # Detection region toggle
        self.detect_full_frame = self.config.get("detect_full_frame", True)
        # Threshold for treating a person as valid (affects both results and annotation)
        self.person_annotation_threshold = self.config.get("person_annotation_threshold", 0.3)
        self.last_crop_offset = (0, 0)

        # Abandonment/Association configuration
        self.abandoned_timeout_seconds = self.config.get("abandoned_timeout_seconds", 30)
        self.association_overlap_threshold = self.config.get("association_overlap_threshold", 0.7)
        self.association_min_duration_seconds = self.config.get("association_min_duration_seconds", 5.0)
        self.suppress_alarm_for_unassociated_items = self.config.get("suppress_alarm_for_unassociated_items", True)

        # Item tracking state
        self.item_tracks = {}
        self.next_item_id = 1
        self.track_max_absence_seconds = 10.0

        self.abandoned_items = {}

        # Display toggles are now handled directly in annotation function

        # Threading components
        self.frame_queue = queue.Queue(maxsize=20)
        self.detection_queue = queue.Queue(maxsize=20)
        self.display_queue = queue.Queue(maxsize=20)

        self.capture_thread = None
        self.detection_thread = None
        self.display_thread = None
        self.monitoring_thread = None
        self.keyboard_thread = None

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
        
        # FPS display
        self.show_fps = self.config["annotations"].get("show_fps", True)
        self.benchmark = Benchmark()
        # Runtime annotation toggles
        self.show_bboxes = True
        self.show_zone = self.config.get("annotations", {}).get("bathroom_zone", True)

        # Statistics
        self.stats = {
            'customers_scanned': 0,
            'merchandise_detected': 0,
            'announcements_made': 0,
            'abandoned_items': 0,
            'theft_deterred': 0
        }

        # Logger setup
        log_file = self.config.get("log_file", "logs/alerts.log")
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self.logger = logging.getLogger("alerts_logger")
            self.logger.setLevel(logging.INFO)
            # Avoid duplicate handlers if multiple instances created
            if not self.logger.handlers:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            self.logger.propagate = False
        except Exception as e:
            print(f"⚠️  Failed to initialize logger: {e}")
            self.logger = None

        # Create images directory for saving alerts
        os.makedirs("images", exist_ok=True)

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
        self.keyboard_thread = threading.Thread(target=self._monitor_keyboard, daemon=True)

        self.capture_thread.start()
        self.detection_thread.start()
        self.display_thread.start()
        self.monitoring_thread.start()
        self.keyboard_thread.start()

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

            # Optionally crop to bathroom zone before detection
            source_frame = frame
            crop_offset = (0, 0)
            if not self.detect_full_frame and frame is not None:
                h, w = frame.shape[:2]
                x1 = int(self.bathroom_zone['x1'] * w)
                y1 = int(self.bathroom_zone['y1'] * h)
                x2 = int(self.bathroom_zone['x2'] * w)
                y2 = int(self.bathroom_zone['y2'] * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    source_frame = frame[y1:y2, x1:x2].copy()
                    crop_offset = (x1, y1)

            # Run YOLO detection on selected region
            results = self.model.predict(
                source_frame,
                imgsz=self.config["imgsz"],
                classes=self.merchandise_classes + self.person_classes,
                conf=self.config["confidence_threshold"],
                verbose=False,
                max_det=self.config["max_detections"]
            )

            # Store current results for monitoring thread
            self.current_results = results
            self.benchmark.update(results[0].speed)

            # Store crop offset for other threads
            self.last_crop_offset = crop_offset

            # Create annotated frame (no tracking)
            annotated_frame = self._annotate_frame(frame, results, crop_offset)

            # Add to display queue
            try:
                self.display_queue.put((annotated_frame, results), timeout=0.01)
            except queue.Full:
                pass  # Skip if display queue is full

    def _display_frames(self):
        """Thread function: Display annotated frames in OpenCV window"""
        cv2.namedWindow('Bathroom Monitor', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('Bathroom Monitor', self.config['window_size'][0], self.config['window_size'][1])

        while self.running:
            try:
                annotated_frame, _ = self.display_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Store for other threads
            self.annotated_frame = annotated_frame

            # Add bathroom zone overlay if enabled
            if self.show_zone:
                self._draw_bathroom_zone(annotated_frame)

            # Add statistics overlay
            self._draw_stats_overlay(annotated_frame)

            if self.show_fps:
                self._draw_fps_overlay(annotated_frame)

            # Display frame
            cv2.imshow('Bathroom Monitor', annotated_frame)

            # Handle key presses for quit/stats and runtime toggles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Signal shutdown; allow main/finalizer to handle cleanup
                self.running = False
                self.stopped = True
                break
            elif key == ord('s'):
                self._print_stats()
            elif key in (ord('f'), ord('F')):
                self.show_fps = not self.show_fps
            elif key in (ord('b'), ord('B')):
                self.show_bboxes = not self.show_bboxes
            elif key in (ord('z'), ord('Z')):
                self.show_zone = not self.show_zone

        # Close the display window from the same thread that created it
        try:
            cv2.destroyWindow('Bathroom Monitor')
        except Exception:
            pass

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
                            # Only add person if confidence above configured threshold
                            if conf >= self.person_annotation_threshold:
                                people_in_zone.append((x1, y1, x2, y2, conf, result))
                            #print(f"DEBUG: Person detected overlapping zone - Class ID: {cls}, Conf: {conf:.2f}")
                        elif cls in self.merchandise_classes:
                            merchandise_in_zone.append((x1, y1, x2, y2, conf, cls, result))

            # Update item tracks and check for abandonment
            try:
                self._update_item_tracks(people_in_zone, merchandise_in_zone)
            except Exception as e:
                print(f"⚠️  Tracking error: {e}")

            # Decide whether to raise alarm based on suppression toggle and track states
            if merchandise_in_zone and self._should_announce():
                should_alarm = True
                if self.suppress_alarm_for_unassociated_items:
                    # Alarm only if any item is currently associated with a person
                    # or has already been marked abandoned
                    should_alarm = False
                    for track in self.item_tracks.values():
                        if track.get('in_zone') and (track.get('associated') or track.get('abandoned_reported')):
                            should_alarm = True
                            break

                if should_alarm:
                    self._make_announcement(len(merchandise_in_zone))
                    self.stats['merchandise_detected'] += len(merchandise_in_zone)
                    self.stats['announcements_made'] += 1
                    self.stats['customers_scanned'] += len(people_in_zone) if people_in_zone else 1

                    if people_in_zone:
                        print(f"🚨 ALARM: {len(merchandise_in_zone)} merchandise items detected with {len(people_in_zone)} people in zone")
                    else:
                        print(f"🚨 ALARM: {len(merchandise_in_zone)} merchandise items detected in zone (no person currently detected)")

                    if self.logger:
                        try:
                            self.logger.info(f"ALERT: items_in_zone count={len(merchandise_in_zone)} people_in_zone={len(people_in_zone)}")
                        except Exception:
                            pass

            time.sleep(1.0)  # Monitor every 1 second to allow audio to complete

    def _update_item_tracks(self, people_in_zone, merchandise_in_zone):
        """Update per-item tracks, manage association with people, and flag abandonment.

        people_in_zone: list of tuples (x1, y1, x2, y2, conf, result)
        merchandise_in_zone: list of tuples (x1, y1, x2, y2, conf, cls, result)
        """
        current_time = time.time()

        # Mark all existing tracks as not updated this cycle
        for track in self.item_tracks.values():
            track['updated_this_cycle'] = False

        # Helper to compute IoU
        def iou(box_a, box_b):
            ax1, ay1, ax2, ay2 = box_a
            bx1, by1, bx2, by2 = box_b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                return 0.0
            area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
            area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
            union = area_a + area_b - inter_area
            return inter_area / union if union > 0 else 0.0

        # Helper to compute fraction of item box overlapped by a person box
        def overlap_fraction_item_by_person(item_box, person_box):
            ix1, iy1, ix2, iy2 = item_box
            px1, py1, px2, py2 = person_box
            inter_x1 = max(ix1, px1)
            inter_y1 = max(iy1, py1)
            inter_x2 = min(ix2, px2)
            inter_y2 = min(iy2, py2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            item_area = max(1, (ix2 - ix1) * (iy2 - iy1))
            return inter_area / item_area

        # Try to match each detected item to an existing track
        for (ix1, iy1, ix2, iy2, iconf, icls, result) in merchandise_in_zone:
            item_box = (int(ix1), int(iy1), int(ix2), int(iy2))

            # Find best matching existing track of same class by IoU
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.item_tracks.items():
                if track['class_id'] != icls:
                    continue
                current_iou = iou(item_box, track['bbox'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= 0.3:
                # Update existing track
                track = self.item_tracks[best_track_id]
                track['bbox'] = item_box
                track['last_seen'] = current_time
                track['updated_this_cycle'] = True
                track['in_zone'] = True
                track['result'] = result
            else:
                # Create new track
                track_id = f"item_{self.next_item_id}"
                self.next_item_id += 1
                self.item_tracks[track_id] = {
                    'id': track_id,
                    'class_id': icls,
                    'bbox': item_box,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'in_zone': True,
                    'associated': False,
                    'association_start_time': None,
                    'unassociated_since': current_time,
                    'abandoned_reported': False,
                    'updated_this_cycle': True,
                    'result': result,
                    'image_saved_for_association': False
                }
                # Log detection of a new item in zone
                if self.logger:
                    try:
                        self.logger.info(f"DETECTED_IN_ZONE: id={track_id} class={icls} bbox={item_box}")
                    except Exception:
                        pass

        # For tracks not updated this cycle, mark absence
        for track_id, track in list(self.item_tracks.items()):
            if not track['updated_this_cycle']:
                # Not seen in this iteration
                # If absent for too long, remove track
                if current_time - track['last_seen'] > self.track_max_absence_seconds:
                    del self.item_tracks[track_id]
                continue

            # Evaluate association with any person based on overlap fraction
            overlapped_now = False
            for (px1, py1, px2, py2, pconf, _) in people_in_zone:
                person_box = (int(px1), int(py1), int(px2), int(py2))
                fraction = overlap_fraction_item_by_person(track['bbox'], person_box)
                if fraction >= self.association_overlap_threshold:
                    overlapped_now = True
                    break

            if overlapped_now:
                # If overlapping, save an image for evidence if not already saved
                result_to_save = track.get('result')
                if result_to_save and not track.get('image_saved_for_association'):
                    now = datetime.now()
                    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
                    image_name = f"img_{timestamp}.jpg"
                    image_path = os.path.join(self.config['images_folder'], image_name)
                    result_to_save.orig_img = cv2.putText(img=result_to_save.orig_img, 
                                                            text=timestamp, org=(20,50),
                                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                            fontScale=2.0, color=(255,255,255),
                                                            thickness=6)
                    cv2.imwrite(image_path, result_to_save.orig_img)
                    track['image_saved_for_association'] = True
                    if self.logger:
                        self.logger.info(f"OVERLAP_DETECTED: Saved image {image_path} for track {track_id}")

                # Start or continue association timing
                if track['association_start_time'] is None:
                    track['association_start_time'] = current_time
                # If persisted long enough, mark associated
                if (current_time - track['association_start_time']) >= self.association_min_duration_seconds:
                    if not track['associated']:
                        track['associated'] = True
                        # Optional log: association achieved
                        if self.logger:
                            try:
                                self.logger.info(f"ASSOCIATED_WITH_PERSON: id={track_id} class={track['class_id']} bbox={track['bbox']}")
                            except Exception:
                                pass
                # Reset unassociated timer
                track['unassociated_since'] = None
            else:
                # No overlap now
                track['association_start_time'] = None
                track['image_saved_for_association'] = False  # Reset image save flag
                if track['associated']:
                    # Lost association once overlap ended
                    track['associated'] = False
                if track['unassociated_since'] is None:
                    track['unassociated_since'] = current_time

            # Check abandonment condition
            if (track['in_zone'] and not track['associated'] and track['unassociated_since'] is not None
                    and not track['abandoned_reported']
                    and (current_time - track['unassociated_since']) >= self.abandoned_timeout_seconds):
                track['abandoned_reported'] = True
                self.stats['abandoned_items'] += 1
                self.stats['theft_deterred'] += 1
                print(f"ALERT: Abandoned merchandise detected (id={track_id}) for {current_time - track['unassociated_since']:.1f}s")
                if self.logger:
                    try:
                        self.logger.info(f"ABANDONED: id={track_id} class={track['class_id']} bbox={track['bbox']} duration={current_time - track['unassociated_since']:.1f}s")
                    except Exception:
                        pass

    def _annotate_frame(self, frame, results, crop_offset=(0, 0)):
        """Annotate frame with detection boxes and labels based on toggle settings"""
        annotated_frame = frame.copy()

        if not self.show_bboxes:
            return annotated_frame

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
                # If detection was on crop, offset back to full-frame coords
                if crop_offset != (0, 0):
                    x1 += crop_offset[0]
                    x2 += crop_offset[0]
                    y1 += crop_offset[1]
                    y2 += crop_offset[1]

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Determine if we should show this detection based on toggles
                should_show = False

                if cls in self.person_classes:
                    # Show person boxes only if confidence exceeds person_annotation_threshold
                    should_show = show_persons and (conf >= self.person_annotation_threshold)

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

    def _monitor_keyboard(self):
        """Background thread to monitor B/F/Z key presses to toggle overlays"""
        while self.running:
            try:
                if msvcrt and msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ('b', 'B'):
                        self.show_bboxes = not self.show_bboxes
                    elif ch in ('f', 'F'):
                        self.show_fps = not self.show_fps
                    elif ch in ('z', 'Z'):
                        self.show_zone = not self.show_zone
            except Exception:
                pass
            time.sleep(0.05)

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

    
    def _draw_fps_overlay(self, frame):
        """Draw FPS overlay on frame"""
        fps = self.benchmark.get_fps()
        fps_text = f"FPS: {fps:.2f}"
        
        # Use user-specified scale factor for consistency
        user_scale = self.stats_scale_factor
        
        # Scaled font properties
        text_scale = max(0.2, 0.7 * user_scale)
        text_thickness = max(1, int(2 * user_scale))
        
        # Position at top-right corner
        (text_width, text_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)
        
        margin = max(5, int(10 * user_scale))
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Position at top right
        text_x = w - text_width - margin
        text_y = text_height + margin
        
        # Draw background rectangle for better visibility
        cv2.rectangle(frame, (text_x - margin, text_y - text_height - margin),
                      (text_x + text_width + margin, text_y + margin), (0, 0, 0), -1)
        
        # Draw FPS text
        cv2.putText(frame, fps_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)


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