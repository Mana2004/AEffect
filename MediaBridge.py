import cv2


class MediaBridge:
    def __init__(self, source_type="webcam", path=0, max_width=1080):
        self.source_type = source_type
        self.static_image = None
        self.max_width = max_width

        if source_type == "webcam":
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Source access failure for identifier: {path}")

            success, test_frame = self.cap.read()
            if not success:
                raise RuntimeError("Empty stream package returned from driver.")
            self.height, self.width = test_frame.shape[:2]

        elif source_type == "photo":
            raw_image = cv2.imread(path)
            if raw_image is None:
                raise RuntimeError(f"Disk read error. Cannot open image at: {path}")

            h, w = raw_image.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                new_w, new_h = int(w * scale), int(h * scale)
                self.static_image = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                print(f"Photo resized from {w}x{h} to {new_w}x{new_h} for performance.")
            else:
                self.static_image = raw_image

            self.height, self.width = self.static_image.shape[:2]

    def get_frame(self):
        if self.source_type == "photo":
            return self.static_image.copy()

        if self.source_type == "webcam":
            success, frame = self.cap.read()
            if success:
                return frame

        return None

    def release(self):
        if self.source_type == "webcam" and hasattr(self, 'cap'):
            self.cap.release()