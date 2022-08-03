from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

import numpy as np


###### Program Parameters ######
error1 = 25
error2 = 30
backup_iterations = 5
################################

###### Testnig Parameters ######
file = 'org/13.tif'
################################


df_org = pd.read_csv('sample.csv')

original_x, original_y = df_org.iloc[:, 0], df_org.iloc[:, 1]
original_points = np.array([original_x, original_y]).T


# read image
image_org = cv.imread(file)
image_org = cv.cvtColor(image_org, cv.COLOR_BGR2GRAY)

X, Y = np.meshgrid(np.arange(-512, 512), np.arange(-512, 512))
image_org[np.sqrt(X**2 + Y**2) > 979/2] = 0


image = cv.medianBlur(image_org, 7)

kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])


image = cv.GaussianBlur(image, (7, 7), 0)

image1 = cv.filter2D(image, -1, kernel)
image2 = cv.filter2D(image, -1, -kernel)
image3 = cv.filter2D(image, -1, kernel.T)
image4 = cv.filter2D(image, -1, -(kernel.T))

image = image1 + image2 + image3 + image4
 
image_org[image > 50] = 0

circles = cv.HoughCircles(image_org, cv.HOUGH_GRADIENT, 1, 20, None, 150, 10, 6, 8)

detected_points = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        detected_points.append((i[0], i[1]))
        cv.circle(image, (i[0], i[1]), i[2], (255, 255, 255), 1)
else:
    print("No circles detected")
    exit()
cv.imshow('image', image)
cv.waitKey(0)

detected_points = np.array(detected_points)
detected_x, detected_y = detected_points[:, 0], detected_points[:, 1]

Match = namedtuple('Match', ['original', 'detected', 'size'])

def expandMatch(match, new, step, original_distance_matrix, detected_distance_matrix):
    for i, j in zip(match.detected, match.original):
        if np.linalg.norm(detected_distance_matrix[i, step] - original_distance_matrix[j, new]) > error2:
            return False
    return True

def getDistanceMatrix(points):
    return np.array([np.linalg.norm(p - points, axis=1) for p in points])


print(f'''
    number of detected points: {len(detected_points)}
    number of original points: {len(original_points)}
    ratio of detected points: {len(detected_points) / len(original_points)}
''')

plt.xlim(0, max(original_x))
plt.ylim(0, max(original_y))
plt.plot(original_x, original_y, 'bo', label='original')
plt.plot(detected_x, detected_y, 'ro', label='detected')
plt.show()


original_distance_matrix = getDistanceMatrix(original_points)
detected_distance_matrix = getDistanceMatrix(detected_points)

n, m = len(detected_points), len(original_points)

matches = []
for i in range(m):
    for j in range(m):
        if np.linalg.norm(detected_distance_matrix[0, 1] - original_distance_matrix[i, j]) < error1:
            matches.append(Match(original=(i, j), detected=(0, 1), size=2))


candidatesInEachStep = [len(matches)]
biggestCandidateInEachStep = [max(match.size for match in matches)]

print('Number of matches in each step:')
print('\t 1. ', len(matches))

for step in range(2, len(detected_points)):
    ###### Update Section ######
    new_matches = []
    for match in matches:
        foundMatch = False
        for i in range(m):
            if expandMatch(match, i, step, original_distance_matrix, detected_distance_matrix):
                new_matches.append(Match(
                    original=(*match.original, i),
                    detected=(*match.detected, step),
                    size=match.size + 1
                ))
                foundMatch = True
        if not foundMatch:
            new_matches.append(match)

    matches = new_matches

    ###### Filter Section ######
    biggestMatch = max(matches, key=lambda match: match.size)

    matches = [
            match for match in matches if match.detected[-1] > step - backup_iterations
        ] + [
            match for match in matches if match.size == biggestMatch.size and match.detected[-1] <= step - backup_iterations
        ]
    ###########################
    
    print(f'\t {step}. ', len(matches))
    candidatesInEachStep.append(len(matches))
    biggestCandidateInEachStep.append(biggestMatch.size)

    # draw_matches = matches
    # for match in draw_matches:
    #     plt.clf()

    #     for i in range(step + 1):
    #         plt.plot(detected_points[i, 0], detected_points[i, 1], 'rx')
    #     plt.plot(detected_points[step + 1, 0], detected_points[step + 1, 1], 'yx')


    #     for i in range(m):
    #         plt.plot(original_points[i, 0], original_points[i, 1], 'go')
        
    #     x = [original_points[i, 0] for i in match.original] + [original_points[match.original[0], 0]]
    #     y = [original_points[i, 1] for i in match.original] + [original_points[match.original[0], 1]]
    #     plt.plot(x, y, 'b-')

    #     plt.xlim(0, max(original_x))
    #     plt.ylim(0, max(original_y))
    #     plt.pause(1 / len(matches))
    # plt.pause(0.1)

print()
print(f'Number of matches: {len(matches)}')
if len(matches) != 0:
    print(f'Biggest candidate: {biggestCandidateInEachStep[-1]}')


bestMatches = [match for match in matches if match.size == biggestCandidateInEachStep[-1]]
print(f'Number of best matches: {len(bestMatches)}')
for match in bestMatches:
    outliers = [i for i in range(n) if i not in match.detected]
    plt.xlim(0, max(original_x))
    plt.ylim(0, max(original_x))
    plt.scatter(detected_x, detected_y, s=15, edgecolors='g', facecolors='none')
    plt.scatter(original_x, original_y, s=12, color='b')
    plt.scatter(detected_points[outliers, 0], detected_points[outliers, 1], s=30, edgecolors='r', facecolors='none')

    for i, j in zip(match.detected, match.original):
        plt.scatter(detected_points[i, 0], detected_points[i, 1], s=12, color='g')
        plt.plot([detected_points[i, 0], original_points[j, 0]], [detected_points[i, 1], original_points[j, 1]], 'g-')
    plt.show()

x = list(range(1, len(candidatesInEachStep) + 1))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, candidatesInEachStep, 'g-')
ax2.plot(x, biggestCandidateInEachStep, 'b-')
plt.show()