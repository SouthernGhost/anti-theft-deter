import time
from collections import deque

class Benchmark:
    """
    A class to benchmark the performance of YOLO object detection, calculating FPS
    for both CPU and GPU devices. It also provides a smoothed FPS for display.
    """
    def __init__(self, device='cpu', window_size=30):
        """
        Initializes the Benchmark instance.
        Args:
            device (str): The device being used for inference ('cpu' or 'gpu').
            window_size (int): The number of recent FPS values to average for a stable display.
        """
        self.device = device
        self.frame_times = deque(maxlen=window_size)
        self.fps = 0
        self.last_update_time = time.time()

    def update(self, speed_data):
        """
        Updates the benchmark with new speed data from a YOLO result.
        Args:
            speed_data (dict): A dictionary from YOLO results with 'preprocess', 'inference',
                               and 'postprocess' times in milliseconds.
        """
        current_time = time.time()
        time_diff = current_time - self.last_update_time

        if time_diff > 0:
            # Calculate total processing time from the speed dictionary
            total_time_ms = speed_data.get('preprocess', 0) + \
                            speed_data.get('inference', 0) + \
                            speed_data.get('postprocess', 0)
            
            # Convert total time to seconds and calculate FPS for this frame
            if total_time_ms > 0:
                frame_fps = 1000 / total_time_ms
                self.frame_times.append(frame_fps)
        
        # Calculate the average FPS over the window
        if self.frame_times:
            self.fps = sum(self.frame_times) / len(self.frame_times)
        
        self.last_update_time = current_time

    def get_fps(self):
        """
        Returns the smoothed frames per second.
        Returns:
            float: The current smoothed FPS.
        """
        return self.fps

    def get_device(self):
        """
        Returns the device being benchmarked.
        Returns:
            str: The device name ('cpu' or 'gpu').
        """
        return self.device.upper()