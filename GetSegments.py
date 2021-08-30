import sys
import numpy as np
import cv2 as cv
import math
from PIL import Image

#Compute the distance from point p to segment [s0, s1].
#Compute also the time (from 0 to 1) when that minimum distance is reached
#t = 0 if minimum distance reached at s0
#t = 1 if minimum distance reached at s1
#otherwise, minimum distance is reached at some point pmin from s0 to s1


def DistToSegmentExtended(s0, s1, p):
    t = 0
    pmin = [0, 0]
    vx = s1[0] - s0[0]
    vy = s1[1] - s0[1]
    a = vx * (p[0] - s0[0]) + vy * (p[1] - s0[1])
    b = vx * vx + vy * vy
    if a <= 0:
        t = 0
        pmin[0] = s0[0]
        pmin[1] = s0[1]
    elif b <= a:
        t = 1
        pmin[0] = s1[0]
        pmin[1] = s1[1]
    else:
        a /= b
        t = a
        pmin[0] = s0[0] + a * vx
        pmin[1] = s0[1] + a * vy
    return [math.sqrt((p[0] - pmin[0]) * (p[0] - pmin[0]) + (p[1] - pmin[1]) * (p[1] - pmin[1])), t]

#Compute distance from point p to segment [s0, s1]
def DistToSegment(s0, s1, p):
    pmin = [0, 0]
    [d, t] = DistToSegmentExtended(s0, s1, p)
    return d

#Return true if and only if point p is within the segment [s0, s1] (with a tolerance of threshold)
#This happens if distance from p to [s0, s1] is within the threshold and t is between 0 and 1
def PointInsideSegment(s0, s1, p, threshold):
    [d, t] = DistToSegmentExtended(s0, s1, p)
    if d > threshold:
        return False
    return t >= 0.0 and t <= 1.0

#Return true if and only if the segment [q0, q1] is inside the segment [s0, s1] (within the threshold)
#This happens if points q0 and q1 are both insides the segment [s0, s1]
def SegmentInsideSegment(s0, s1, q0, q1, threshold):
    return PointInsideSegment(s0, s1, q0, threshold) and PointInsideSegment(s0, s1, q1, threshold)

#Add the segment [q0, q1] to segments only if [q0, q1] is not contained by any segment in Segments
#Also, remove all segments in segs that are contained inside [q0, q1]
def UpdateSegments(segs, q0, q1, threshold):
    #determine if [q0, q1] is contained inside some segment [s0, s1] in segs
    #if so, no need to do anything
    n = len(segs)
    for i in range(0, n, 2):
        s0 = segs[i]
        s1 = segs[i + 1]
        if SegmentInsideSegment(s0, s1, q0, q1, threshold):
            return segs
    #remove all segments in segs that are contained inside [q0, q1]
    i = 0
    while i < len(segs):
        n = len(segs)
        s0 = segs[i]
        s1 = segs[i + 1]
        if SegmentInsideSegment(q0, q1, s0, s1, threshold):
            segs[i] = segs[n - 2]
            segs[i + 1] = segs[n - 1]
            segs = segs[0:-2]
        else:
            i = i + 2
    #add segment [q0, q1]
    segs.append(q0)
    segs.append(q1)


#Determine if points from start to end fit within the segment [s0, s1]
def DoPointsFitSegment(s0, s1, pts, start, end, threshold):
    if start > end:
        return True
    for i in range(start, end + 1):
        d = DistToSegment(s0, s1, pts[i])
        if d > threshold:
            #print(f'dist {d} at {i} from {start} to {end}')
            return False
    return True

#Build segments along the sequence of points
#Try to extend the segments as much as possible
def BuildSegments(pts, threshold):
    (nrRows, nrCols) = pts.shape
    if nrRows == 1:
        return [pts[0], pts[0]]
    segs = []
    i = 0
    while i < nrRows:
        j = nrRows - 1
        while j > i:
            if DoPointsFitSegment(pts[i], pts[j], pts, i + 1, j - 1, threshold):
                segs.append(pts[i])
                segs.append(pts[j])
                break
            j = j - 1
        if j > i:
            i = j
        else:
            i = i + 1
    return segs

def DrawSegments(img, segs, rgb, thick):
    n = len(segs)
    for i in range(0, n, 2):
        ps = (segs[i][0], segs[i][1])
        pe = (segs[i+1][0], segs[i+1][1])
        img = cv.line(img, ps, pe, rgb, thick)

def main():

    file_name = sys.argv[1]
    b_matrix = np.loadtxt(file_name, dtype=int)
    rows, cols = b_matrix.shape
    for i in range(rows):
        for j in range(cols):
            if b_matrix[i][j] == 1:
                b_matrix[i][j] = 0
            else:
                b_matrix[i][j] = 1
    b_matrix = np.array(b_matrix).astype(np.uint8)
    pil_image = Image.fromarray((b_matrix * 255))
    pil_image.save('boundary_photo.jpg', 'JPEG')

    im = cv.imread('boundary_photo.jpg') # <--- GIVE THE PATH TO A BINARY IMAGE
    imConts = im.copy()
    imSegs = im.copy()
    imSegsAll = im.copy()
    cv.imshow('original', im)
    im2 = im.copy()
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(imConts, contours, -1, (0, 0, 255), 1)
    cv.imshow('Contours', imConts)

    others = 0
    threshold = 3
    csegs = []
    all_segs = []
    count = 0
    for c in contours:
        pts = c.reshape((c.shape[0], c.shape[2]))
        segs = BuildSegments(pts, threshold)
        others = others + len(segs)
        csegs.append(segs)
        DrawSegments(imSegs, segs, (255, 0, 0), 2)
        print(f'countour {count}')
        print(pts)
        print(f'segments for contour {count}')
        for p in segs:
            print(p)
        count = count + 1
        for k in range(0, len(segs), 2):
            UpdateSegments(all_segs, segs[k], segs[k+1], threshold)
    cv.imshow('Segments', imSegs)

    DrawSegments(imSegsAll, all_segs, (0, 0, 0), 2)
    cv.imshow('AllSegments', imSegsAll)

    print(f'all segs = {len(all_segs)}')
    print(f'others = {others}')
    cv.waitKey(0)

if __name__ == "__main__":
    main()
