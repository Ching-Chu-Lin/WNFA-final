import datetime
from decode import ROW_NUM
import imutils
from threading import Thread
import threading
import cv2
import time
import numpy as np


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
        cnt += 1
    # cv2.destroyAllWindows()


def Process(chunks):
    if len(chunks) >= 900:
        print(chunks[0].shape)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280,  720))
        for i in chunks:
            out.write(i)
        out.release()
        exit()

def get_difference(previous_frame, current_frame):
    previous_frame = previous_frame.astype(np.int32)
    current_frame = current_frame.astype(np.int32)

    d_pixel = np.sum(np.abs(previous_frame - current_frame))/(previous_frame.shape[0]*previous_frame.shape[1])
    old_bins = np.zeros(766)
    new_bins = np.zeros(766)
    for i in range(previous_frame.shape[0]):  # height
        for j in range(previous_frame.shape[1]):  # width
            old_bins[np.sum(previous_frame[i, j, :])] += 1
            new_bins[np.sum(current_frame[i, j, :])] += 1
    d_histo = np.nansum(np.power((old_bins - new_bins), 2)/np.maximum(old_bins, new_bins))

    return d_pixel, d_histo


def decode(chunks, c_lock):
    chunks_ind = 0
    currentFrame = None
    nextFrame = None
    # constant for determining scene type
    d_pixel_cut = 100
    d_pixel_static = 10
    d_histo_cut = 1
    d_histo_static = 0.1
    while True:
        enough_frame = False
        with c_lock:
            if len(chunks)-chunks_ind > 12:
                currentFrameWindow = chunks[chunks_ind:chunks_ind+6]
                nextFrameWindow = chunks[chunks_ind+6:chunks_ind+12]
                enough_frame = True
                chunks_ind += 6
        if enough_frame:
            # read current and next window (total of 12 frames)
            # currentFrameWindow = read_window()
            # nextFrameWindow = read_window(vidcap)
            currentSampleFrame = currentFrameWindow[0]
            nextSampleFrame = nextFrameWindow[0]

            height = currentSampleFrame.shape[0]
            width = currentSampleFrame.shape[1]

            # resized_current_image = resize_image(currentSampleFrame)
            # resized_next_image = resize_image(nextSampleFrame)

            # get difference
            resized_current_frame = cv2.resize(currentSampleFrame, (int(width/4), int(height/4)))
            resized_next_frame = cv2.resize(nextSampleFrame, (int(width/4), int(height/4)))
            d_pixel, d_histo = get_difference(resized_current_frame, resized_next_frame)

            # determine frame scene
            if d_pixel > d_pixel_cut and d_histo > d_histo_cut:  # cut scene
                return
                # frameScene = cut
                # currentSampleFrame = nextSampleFrame # assign the new frame to previous frame for next window
                # currentFrameWindow = nextFrameWindow # assign the next window to current window
                # continue
            # elif d_pixel < d_pixel_static and d_histo < d_histo_static: # static scene
            #     frameScene = static
            # else:
            #     frameScene = gradual

            #  sum RGB
            currentFrameWindow = np.sum(currentFrameWindow, axis=3)

            # display pixel value
            # fprintf("--------------------\npixel value on %d th frame\n", i-6)
            # fprintf("%d %d %d %d %d %d\n", currentFrameWindow(360, 640, 1), currentFrameWindow(360, 640, 2), currentFrameWindow(360, 640, 3), currentFrameWindow(360, 640, 4), currentFrameWindow(360, 640, 5), currentFrameWindow(360, 640, 6))

            # cut into grids
            ROW_NUM = 1
            COL_NUM = 1
            row_num = ROW_NUM
            col_num = COL_NUM
            for j in range(row_num):
                for k in range(col_num):
                    center_height = int((j+0.5)*height/row_num)
                    center_width = int((k+0.5)*width/col_num)

                    # pixelIntensity = (squeeze(currentFrameWindow(centerHeight, centerWidth, : ))).'
                    pixel_intensity = currentFrameWindow[:, center_height, center_width]
                    alpha_value = 1 - pixel_intensity/pixel_intensity[1]

                    # extended_alpha = np.tile(alphaValue, 5)
                    demodulation = np.fft.fft(alpha_value, 30)
                    # Pyy = demodulation * np.conjugate(demodulation) / 30 # normalization
                    Pyy = abs(demodulation) / 30

                    # freq = np.arange(0, 15)*60/30
                    # plt.plot(freq, Pyy[:15])
                    # plt.show()

                    if Pyy[11] > Pyy[16]:
                        print("0")
                        # output_value.append(0)
                    else:
                        print("1")
                        # output_value.append(1)

    # print(len(chunks))


def main():
    frames = []
    frames_ind = 0
    fps = 30
    second = None
    chunks = []
    chunk_len = 6
    lock = threading.Lock()
    c_lock = threading.Lock()
    thread1 = threading.Thread(target=capture, args=(frames, lock))
    thread1.start()
    thread2 = threading.Thread(target=capture, args=(frames, lock))
    thread2.start()
    thread3 = threading.Thread(target=decode, args=(chunks, c_lock))
    thread3.start()
    # capture(frames)
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
            # Process(chunks)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
