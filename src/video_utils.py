import cv2, os
from collections import deque

class VideoSource:

    def __init__(self, filename, n_slices=2):
        
        self.filename = filename
        if not os.path.exists( filename ):
            raise FileNotFoundError
        
        self.cap = cv2.VideoCapture(filename)
        self.vid = deque(maxlen=n_slices)

        while len(self.vid) < n_slices-1:
            success, frame = self.cap.read()
            if not success:
                raise ValueError('Unable to read form the video file')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.vid.append( frame )

    def __iter__(self):
        return self
    
    def __next__(self):
        success, frame = self.cap.read()
        if not success:
            raise StopIteration
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.vid.append( frame )
        return self.vid
    
