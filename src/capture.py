import datetime
import imutils
from threading import Thread
import threading
import cv2
import time
def capture(frames, lock):
    stream = cv2.VideoCapture(0)
    cnt = 0
    while cnt < 990:
        (grabbed, frame) = stream.read()
        with lock:
            frames.append(frame)
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
        if cnt == 0:
            cv2.imwrite("initial.png", frame)
        cnt+=1
    #cv2.destroyAllWindows()
def Process(chunks):
    if len(chunks) >= 900:
        print(chunks[0].shape)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280,  720))
        for i in chunks:
            out.write(i)
        out.release()
        exit()
def decode(chunks, c_lock):
    chunks_ind = 0
    currentFrame = None
    nextFrame = None
    while True:
        enough_frame = False
        with c_lock:
            if len(chunks)-chunks_ind > 12:
                currentFrame = chunks[chunks_ind:chunks_ind+6]
                nextFrame = chunks[chunks_ind+6:chunks_ind+12]
                enough_frame = True
                chunks_ind+=6
        if enough_frame:

                
            

    #print(len(chunks))
def main():
    frames = []
    frames_ind = 0
    fps = 30
    second = None
    chunks = []
    chunk_len = 6
    lock = threading.Lock()
    c_lock = threading.Lock()
    thread1 = threading.Thread(target = capture, args=(frames, lock))
    thread1.start()
    thread2 = threading.Thread(target = capture, args=(frames, lock))
    thread2.start()
    thread3 = threading.Thread(target = decode, args=(chunks, c_lock))
    thread3.start()
    #capture(frames)
    time.sleep(1)
    with lock:
        frames_ind = len(frames)
    start_time = time.time()
    try:
        while True:
            if time.time() - start_time >= 1:
                start_time = time.time()
                with lock:
                    second = frames[frames_ind:]
                    frames_ind = len(frames)
                second = [second[int(i*len(second) / fps)] for i in range(fps)]
                chunks.extend(second)
                print(len(chunks))
            #Process(chunks)
    except KeyboardInterrupt:
        pass
if __name__ == "__main__":
    main()

