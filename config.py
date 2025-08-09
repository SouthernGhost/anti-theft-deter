PERSON_IDS_OIV7 = [63, 216, 322, 381, 594]
MERCH_IDS_OIV7 = [10, 16, 17, 21, 39, 57, 65, 67, 72, 76, 78, 82, 84, 85, 86, 89, 92,
                    93, 96, 105, 108, 119, 120, 121, 125, 126, 132, 133, 135, 139, 140,
                    143, 146, 151, 154, 166, 171, 172, 177, 178, 182, 185, 186, 199, 200,
                    204, 207, 210, 211, 213, 217, 219, 223, 226, 227, 229, 233, 237, 238,
                    244, 246, 256, 258, 273, 278, 286, 287, 289, 292, 293, 295, 296, 299,
                    301, 304, 306, 314, 318, 323, 328, 332, 333, 338, 339, 344, 345, 347,
                    351, 356, 365, 367, 368, 372, 373, 374, 375, 377, 378, 380, 385, 386,
                    389, 390, 391, 392, 394, 395, 396, 399, 400, 404, 406, 407, 419, 420,
                    423, 430, 431, 432, 433, 434, 436, 438, 441, 445, 448, 449, 451, 459,
                    461, 468, 475, 480, 481, 483, 486, 495, 496, 501, 503, 505, 507, 516,
                    517, 518, 521, 523, 524, 526, 528, 529, 535, 537, 539, 540, 542, 545,
                    547, 550, 559, 560, 562, 565, 566, 570, 571, 573, 575, 576, 577,
                    579, 584, 589, 590, 591, 592, 593, 595, 598]

PERSON_IDS_COCO = [0]
MERCH_IDS_COCO = [24, 25, 26, 27, 28, 39, 40, 41, 42, 43, 44, 45,
                    63, 64, 65, 66, 67, 73, 74, 76, 77, 78, 79]

CONFIG = {
    # Model Configuration
    "model_path": "yolo11n.pt",  # Path to YOLO model

    # Video Source Configuration
    "stream_mode": True,  # Set to True to use IP camera stream
    "video_source": "videos/vid13.mp4",  # Video file path, webcam (0), or IP camera URL
    "ip_camera_url": "http://192.168.10.9:8080/video",  # IP camera URL (used when stream_mode=True)

    # Detection Zone Configuration
    "bathroom_zone": {
        'x1': 0.01, 'y1': 0.5,   # Top-left corner (relative coordinates 0-1)
        'x2': 0.99, 'y2': 0.99   # Bottom-right corner (relative coordinates 0-1)
    },

    # Display Configuration
    "show_stats": False,  # Show statistics overlay
    "stats_scale_factor": 0.2,  # Scale factor for UI elements (0.1-1.0)

    # Annotation Toggles - Organized by category
    "annotations": {
        "bathroom_zone": True,     # Show/hide bathroom zone rectangle and label
        "persons": True,           # Show/hide person bounding boxes and labels
        "items": True,             # Show/hide item/merchandise bounding boxes and labels
    },

    # Detection Classes
    "merchandise_classes": MERCH_IDS_COCO,

    "person_classes": PERSON_IDS_COCO,  # Person class IDs - add more if needed

    # Stream Configuration
    "max_reconnect_attempts": 10,
    "reconnect_delay": 5,  # seconds

    # Detection Configuration
    "confidence_threshold": 0.1,
    "max_detections": 25,
    "imgsz": 640,

    # Abandonment/Association Configuration
    # Time an item must remain in the zone without being with a person to be considered abandoned
    "abandoned_timeout_seconds": 5,
    # Overlap threshold (fraction of item box area overlapped by a person box) to consider them overlapping
    "association_overlap_threshold": 0.3,
    # Continuous time the overlap must persist to consider the item as with a person
    "association_min_duration_seconds": 0.0,
    # If True, do not raise alarm for items in zone unless they are associated with a person
    # or have exceeded the abandonment timeout. If False, alarm on any item in zone.
    "suppress_alarm_for_unassociated_items": True,

    # Logging configuration
    "log_file": "logs/alerts.log"
}