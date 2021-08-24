import cv2 as cv
import time
import numpy as np
from numpy_ringbuffer import RingBuffer
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json
import scipy.signal

GRIDIRON = []
FRAMEWINDOWNAME = "AutoCutMarks"
DEBUG = True


def clicked(event, x, y, flags, param):
    global GRIDIRON
    if event == cv.EVENT_LBUTTONUP:
        GRIDIRON.append((x,y))
        text = "x: " + str(x) + " y: " + str(y)
        print(text)

def setGridIron(videofile):
    global GRIDIRON
    cap = cv.VideoCapture(videofile)
    fps = cap.get(cv.CAP_PROP_FPS)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    cv.namedWindow(FRAMEWINDOWNAME)
    cv.setMouseCallback(FRAMEWINDOWNAME, clicked)

    # Maybe only use every 2nd or 3rd fame
    while cap.isOpened():
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        showed_frame = frame2.copy()

        if len(GRIDIRON) >= 2:
            pts = np.array(GRIDIRON)
            #pts = pts - pts.min(axis=0)
            mask1= np.zeros(frame1.shape[:2], np.uint8)
            mask2= np.zeros(frame2.shape[:2], np.uint8)
            cv.drawContours(mask1, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
            cv.drawContours(mask2, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
            frame1 = cv.bitwise_and(frame1, frame1, mask=mask1)
            frame2 = cv.bitwise_and(frame2, frame2, mask=mask2)
            #cv.rectangle(showed_frame, (x,y), (x+w, y+h), (255,0,0), 2)
            #pts = pts.reshape((-1,1,2))
            cv.polylines(showed_frame, [pts], True, (255, 0, 0))

        cv.imshow(FRAMEWINDOWNAME, showed_frame)
        if cv.waitKey(int(1000/fps)) == ord('q'):
            break
        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
    cv.destroyAllWindows()

    gridiron = GRIDIRON
    GRIDIRON = []
    return gridiron


def plot_motion_graph(motionGraph, debug):
    x_coords, y_coords = zip(*motionGraph)
    plt.plot(x_coords, y_coords, '-')
    #if debug:
    #    plt.pause(0.0005)
    #else: 
    #    plt.show()


def analyze_motion_graph(motionGraph, snap_threshold_percentage):
    frame_rate = 30

    x_coords, y_coords = zip(*motionGraph)
    #ring = RingBuffer(capacity=10)
    #for (frame, motions) in motionGraph[1:-2]:
        #ring.append(motions)
        #print(ring)
        #if len(ring) > 8:
        #   if ring[-1] - ring[0] > 7:
        #        print("############SNAP")
    
    #coefficients = np.polyfit(x_coords,y_coords,50)
    #poly = np.poly1d(coefficients)
    #new_x = np.linspace(x_coords[0], x_coords[-1])
    #new_y = poly(new_x)
    #plt.plot(new_x, new_y)

    # smoothing the measurements
    filter_window = int(0.5 * frame_rate) # 0.5s * frame rate
    if filter_window % 2 == 0:
        filter_window = filter_window + 1 # only odd filter windows allowed for savgol
    savgol_polynom_degree = 3
    smoothed_y = scipy.signal.savgol_filter(y_coords, filter_window, savgol_polynom_degree)

    firstDeriviative = []
    assert(len(x_coords) == len(smoothed_y))
    for i in range(1, len(smoothed_y)-filter_window):
        motionsA = smoothed_y[i]
        motionsB = smoothed_y[i+filter_window]
        increase = motionsB - motionsA / 1.0
        #print(increase)
        firstDeriviative.append((x_coords[i], increase))
    
    _, firstDeriviative_y = zip(*firstDeriviative)
    max_increase = np.max(firstDeriviative_y)
    snaps = []
    for (x,y) in firstDeriviative:
        if y >= max_increase * snap_threshold_percentage:
            if len(snaps) == 0 or (x - snaps[-1] > 3*frame_rate):  # there can be no two snaps within 3 seconds
                print("snap at frame", x)
                snaps.append(x)
    #for i in range(1, len(motionGraph)-1):
    #    (_, motionsA) = motionGraph[i-1]
    #    (_, motionsB) = motionGraph[i+1]
    #    increase = motionsB - motionsA
    #    print(increase)
    #    firstDeriviative.append((i, increase))
    #plot_motion_graph(motionGraph, debug=False)
    plt.plot(x_coords, smoothed_y)
    plot_motion_graph(firstDeriviative, debug=False)
    plt.show()

    return snaps
    

        


def generateMotionGraph(videofile, gridiron_near, gridiron_far):
    cap = cv.VideoCapture(videofile)
    total_number_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    motionGraph = []

    if len(gridiron_near) < 3 or len(gridiron_far) < 3:
        print("WARNING: gridiron not set properly")

    # Maybe only use every 2nd or 3rd fame
    while cap.isOpened():#json.dump(motionGraph, f)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        showed_frame = frame1.copy()

        motions = []
        for (gridiron, minimum_contour_size) in [(gridiron_near, 900), (gridiron_far, 500)]:
            frameA = frame1.copy()
            frameB = frame2.copy()
            if len(gridiron) >= 3:
                pts = np.array(gridiron)
                mask1= np.zeros(frameA.shape[:2], np.uint8)
                mask2= np.zeros(frameB.shape[:2], np.uint8)
                cv.drawContours(mask1, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
                cv.drawContours(mask2, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
                frameA = cv.bitwise_and(frameA, frameA, mask=mask1)
                frameB = cv.bitwise_and(frameB, frameB, mask=mask2)

            diff = cv.absdiff(frameA, frameB)
            gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray,(5,5), 0)
            _, threshold = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(threshold, None, iterations=3)
            contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


            for contour in contours:
                if cv.contourArea(contour) < minimum_contour_size:
                    continue
                motions.append(contour)
                (x,y, w,h) = cv.boundingRect(contour)
                cv.rectangle(showed_frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
        current_frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)
        motionGraph.append((int(current_frame_number), len(motions)))

        if current_frame_number % 100 == 0 and len(motionGraph) > 1:
            pass #plot_motion_graph(motionGraph, debug=True)
        else:
            text = "Motions: " + str(len(motions))
            cv.putText(showed_frame, text, (10,20), cv.FONT_HERSHEY_SIMPLEX,1, (0,0, 255), 3)

        if DEBUG:
          cv.imshow(FRAMEWINDOWNAME, showed_frame)
          if cv.waitKey(1) == ord('q'):
            break
          #plot_motion_graph(motionGraph, debug=True)
        #if snapped:
        #    time.sleep(0.3)
        frame1 = frame2
        ret, frame2 = cap.read()
        frameNo = cap.get(cv.CAP_PROP_POS_FRAMES)
        if frameNo % 100 == 0:
            print("analyzed frame " + str(frameNo) + " of " + str(total_number_frames))
        #if frameNo > 2000:
        #    break

    cap.release()
    cv.destroyAllWindows()
    return motionGraph




def test_analyzed_cutmarks(videofile, cutmarks):
    cv.namedWindow(FRAMEWINDOWNAME)
    cap = cv.VideoCapture(videofile)
    fps = cap.get(cv.CAP_PROP_FPS)
    for (begin, end) in cutmarks:
        cap.set(cv.CAP_PROP_POS_FRAMES, begin-1)
        current_position = begin-1
        while current_position < end:
            ret, frame = cap.read()
            current_position  = cap.get(cv.CAP_PROP_POS_FRAMES)
            if ret and cap.isOpened():
                cv.imshow(FRAMEWINDOWNAME, frame)
                if cv.waitKey(int(fps)) == ord('q'):
                    break
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break


def calculateCutMarks(snaps):
    fps = 30

    cutmarks = []
    for snap in snaps:
        begin = max(0, snap-2*fps)
        end = snap + 10*fps
        cutmarks.append((begin, end))

    return cutmarks
    

##########################
# main
##########################
videofile = "blub.mp4"

#grid_iron_near = setGridIron(videofile)
grid_iron_near = [
    (712, 164),
    (712 , 164),
    (244 , 341),
    (8 , 492),
    (73 , 731),
    (1738 , 772),
    (1825 , 566),
    (1120 , 171),
 ]

#grid_iron_far = setGridIron(videofile)
grid_iron_far = [
    (722, 165),
    (523 , 237),
    (1303 , 259),
    (1143 , 174),
 ]

load_test_graph = False
if load_test_graph:
    f = open("motionGraph.json", "r")
    motionGraph = json.load(f)
else:
    f = open("motionGraph.json", "w")
    motionGraph  = generateMotionGraph(videofile, grid_iron_near, grid_iron_far)
    json.dump(motionGraph, f)
f.close

snaps = analyze_motion_graph(motionGraph,0.5)
#snaps = [
#1116,   # G
#2062,   #M P1
#6941,   
#7876,   
#8875,   
#15687,  
#16893,  
#18120,  
#18379,  
#19413,  
#19592,  
#22586,  
#24485,  
#26385,  
#27320,  
#30462,  
#31299,  
#32254,  
#38207,  
#39319,  
#40430,  
#42087,  
#44410,  
#45760,  
#46705
#]  

cutmarks = calculateCutMarks(snaps)

test_analyzed_cutmarks(videofile, cutmarks)




