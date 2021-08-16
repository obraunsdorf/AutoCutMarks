import cv2 as cv
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


GRIDIRON = []
FRAMEWINDOWNAME = "AutoCutMarks"


def clicked(event, x, y, flags, param):
    global GRIDIRON
    if event == cv.EVENT_LBUTTONUP:
        GRIDIRON.append((x,y))
        text = "x: " + str(x) + " y: " + str(y)
        print(text)

def setGridIron(videofile):
    cap = cv.VideoCapture("test.mp4")
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


def plot_motion_graph(motionGraph, debug):
    x_coords, y_coords = zip(*motionGraph)
    #coefficients = np.polyfit(x_coords,y_coords,7)
    #poly = np.poly1d(coefficients)
    #new_x = np.linspace(x_coords[0], x_coords[-1])
    #new_y = poly(new_x)
    plt.plot(x_coords, y_coords, '-')
    if debug:
        plt.pause(0.0005)
    else: 
        plt.show()


def findCutMarks(videofile, gridiron):
    cap = cv.VideoCapture(videofile)
    fps = cap.get(cv.CAP_PROP_FPS)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    motionGraph = []


    # Maybe only use every 2nd or 3rd fame
    while cap.isOpened():
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if len(gridiron) >= 3:
            pts = np.array(gridiron)
            mask1= np.zeros(frame1.shape[:2], np.uint8)
            mask2= np.zeros(frame2.shape[:2], np.uint8)
            cv.drawContours(mask1, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
            cv.drawContours(mask2, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
            frame1 = cv.bitwise_and(frame1, frame1, mask=mask1)
            frame2 = cv.bitwise_and(frame2, frame2, mask=mask2)

        showed_frame = frame1

        diff = cv.absdiff(frame1, frame2)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray,(5,5), 0)
        _, threshold = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(threshold, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        motions = []

        snapped = False

        for contour in contours:
            if cv.contourArea(contour) < 900:
                continue
            motions.append(contour)
            (x,y, w,h) = cv.boundingRect(contour)
            cv.rectangle(showed_frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        current_frame_number = cap.get(cv.CAP_PROP_POS_FRAMES)
        motionGraph.append((current_frame_number, len(motions)))
        if current_frame_number % 100 == 0 and len(motionGraph) > 1:
            pass#plot_motion_graph(motionGraph, debug=True)
        if len(motions) < 2:
            cv.putText(showed_frame, "SNAP!", (50,20), cv.FONT_HERSHEY_SIMPLEX,1, (0,0, 255), 3)
            snapped = True
        else:
            text = "Motions: " + str(len(motions))
            cv.putText(showed_frame, text, (10,20), cv.FONT_HERSHEY_SIMPLEX,1, (0,0, 255), 3)

        cv.imshow(FRAMEWINDOWNAME, showed_frame)
        #if snapped:
        #    time.sleep(0.3)
        


        if cv.waitKey(1) == ord('q'):
            break
        frame1 = frame2
        ret, frame2 = cap.read()

    cap.release()
    cv.destroyAllWindows()
    plot_motion_graph(motionGraph, debug=False)




##########################
setGridIron("test.mp4");
#test_grid_iron = [
#    (712, 164),
#    (712 , 164),
#    (244 , 341),
#    (8 , 492),
#    (73 , 731),
#    (1738 , 772),
#    (1825 , 566),
#    (1120 , 171),
#]
findCutMarks("test.mp4", test_grid_iron)



