import cv2
import threading
from queue import Queue


class Camera:
    def __init__(self, source, desired_fps):
        self.source = source
        self._fps = desired_fps
        self._frames_queue = Queue()
        self._exit_signal = threading.Event()

    def start_producer_thread(self):
        def producer_thread():
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FPS, self._fps)

            while not self._exit_signal.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                self._frames_queue.put(frame)

            cap.release()

        producer = threading.Thread(target=producer_thread)
        producer.start()

    def run(self):
        self.start_producer_thread()

    def stop_threads(self):
        self._exit_signal.set()

    @property
    def frames_queue(self):
        return self._frames_queue

    @property
    def exit_signal(self):
        return self._exit_signal