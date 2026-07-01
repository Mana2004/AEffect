import cv2


class MediaBridge:
    def __init__(self, source_type="webcam", path=0):
        self.source_type = source_type
        self.static_image = None

        if source_type == "webcam" or source_type == "video":
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Source access failure for identifier: {path}")
            success, test_frame = self.cap.read()
            if not success:
                raise RuntimeError("Empty stream package returned from driver.")
            self.height, self.width = test_frame.shape[:2]
        elif source_type == "photo":
            self.static_image = cv2.imread(path)
            if self.static_image is None:
                raise RuntimeError(f"Disk read error at address: {path}")
            self.height, self.width = self.static_image.shape[:2]

    def get_frame(self):
        if self.source_type == "photo":
            return self.static_image.copy()

        success, frame = self.cap.read()
        if success:
            return frame
        elif self.source_type == "video":
            # Rewind video clip in a recursive circle loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.cap.read()
            return frame
        return None

    def release(self):
        if self.source_type in ["webcam", "video"] and hasattr(self, 'cap'):
            self.cap.release()