from collections import namedtuple
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-image', help='path to input image', required=True)
parser.add_argument('--outdir', help='path to output image', default='./images')
parser.add_argument('--reference', help='path to reference csv file', default='reference.csv')
parser.add_argument('--save', help='save output image', action='store_true')
parser.add_argument('--no-plot', help='do not plot', action='store_true')
parser.add_argument('--error', help='error threshold', default=25, type=int)
parser.add_argument('--backup-iterations', help='number of backup iterations', default=0, type=int)

args = parser.parse_args()

error1 = args.error
error2 = args.error
backup_iterations = args.backup_iterations

file_path = args.input_image
file_name, file_extension = os.path.splitext(file_path.split('/')[-1])


df_org = pd.read_csv('reference.csv', index_col=0)

original_x, original_y = df_org.iloc[:, 0], df_org.iloc[:, 1]
original_points = np.array([original_x, original_y]).T


image_org = cv.imread(file_path)

### IMAGE NUMBER 57 THROWS AN ERROR AND HAVE TO BE RESIZED ###
image_org = cv.resize(image_org, (1024, 1024))
image_org = cv.cvtColor(image_org, cv.COLOR_BGR2GRAY)

X, Y = np.meshgrid(np.arange(-512, 512), np.arange(-512, 512))
image_org[np.sqrt(X**2 + Y**2) > 979/2] = 0

if not args.no_plot:
    cv.imshow(file_name, cv.resize(image_org, (512, 512)))
    cv.waitKey(0)

image = cv.medianBlur(image_org, 7)

kernel = np.array([
    [1, 0, -1],
    [3, 0, -3],
    [1, 0, -1]
])


image = cv.GaussianBlur(image, (7, 7), 0)

image1 = cv.filter2D(image, -1, kernel)
image2 = cv.filter2D(image, -1, -kernel)
image3 = cv.filter2D(image, -1, kernel.T)
image4 = cv.filter2D(image, -1, -(kernel.T))

image = image1 + image2 + image3 + image4
 
image_org[image > 50] = 0

### Optimal Parameters are 150 and 10 ###
circles = cv.HoughCircles(image_org, cv.HOUGH_GRADIENT, 1, 20, None, 100, 4, 6, 8)

detected_points = []
show_image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        detected_points.append((i[0], i[1]))
        cv.circle(show_image, (i[0], i[1]), i[2], (0, 0, 255), 1)
else:
    print("No circles detected")
    exit()

if not args.no_plot:
    cv.imshow('Processed image', cv.resize(show_image, (512, 512)))
    cv.waitKey(0)
    cv.destroyAllWindows()
if args.save:
    cv.imwrite(os.path.join(args.outdir, file_name + '-processed.png'), cv.resize(show_image, (512, 512)))

detected_points = np.array(detected_points, dtype='float32')
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

fig, ax = plt.subplots()

ax.set_xlim(0, max(original_x))
ax.set_ylim(0, max(original_y))
ax.plot(original_x, original_y, 'bo', label='original')
ax.plot(detected_x, detected_y, 'ro', label='detected')
ax.legend()
if not args.no_plot:
    plt.show()

original_distance_matrix = getDistanceMatrix(original_points)
detected_distance_matrix = getDistanceMatrix(detected_points)

n, m = len(detected_points), len(original_points)

matches = []
for i in range(m):
    for j in range(m):
        if abs(detected_distance_matrix[0, 1] - original_distance_matrix[i, j]) < error1:
            matches.append(Match(original=(i, j), detected=(0, 1), size=2))

candidatesInEachStep = [len(matches)]
biggestCandidateInEachStep = [max(match.size for match in matches)]

print('Number of matches in each step:')
print('\t 1. ', len(matches), biggestCandidateInEachStep[0])

start_time = time.time()
for step in range(2, len(detected_points)):
    ###### Update Section ######
    new_matches = []
    for match in matches:
        foundMatch = False
        for i in range(m):
            if i not in match and expandMatch(match, i, step, original_distance_matrix, detected_distance_matrix):
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


    ## count matches of each size  
    candidatesInEachStep.append(len(matches))
    biggestCandidateInEachStep.append(biggestMatch.size)

    print(f'\t {step}. ', len(matches), biggestCandidateInEachStep[-1])

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

end_time = time.time()
print()
print(f'Number of matches: {len(matches)}')
if len(matches) != 0:
    print(f'Biggest candidate: {biggestCandidateInEachStep[-1]}')


bestMatches = [match for match in matches if match.size == biggestCandidateInEachStep[-1]]
print(f'Number of best matches: {len(bestMatches)}')
print(f'Time: {end_time - start_time}')

if len(bestMatches) == 0:
    print('No matches found')
    exit(-1)
elif len(bestMatches) > 1:
    print('Warning: multiple matches found')

for match in bestMatches[:3]:
    outliers = [i for i in range(n) if i not in match.detected]
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(original_x))
    ax.set_ylim(0, max(original_x))
    ax.scatter(detected_x, detected_y, s=15, edgecolors='g', facecolors='none')
    ax.scatter(original_x, original_y, s=12, color='b')
    ax.scatter(detected_points[outliers, 0], detected_points[outliers, 1], s=30, edgecolors='r', facecolors='none')

    for i, j in zip(match.detected, match.original):
        ax.scatter(detected_points[i, 0], detected_points[i, 1], s=12, color='g')
        ax.plot([detected_points[i, 0], original_points[j, 0]], [detected_points[i, 1], original_points[j, 1]], 'g-', linewidth=1)
    

    if not args.no_plot:
        plt.show()
    if args.save:
        fig.savefig(os.path.join(args.outdir, file_name + '-matching.png'))

    ### Find transformation matrix ###
    center_original = np.mean(original_points[match.original, :], axis=0)
    center_detected = np.mean(detected_points[match.detected, :], axis=0)
    A = np.zeros((2, 2))
    for i in range(match.size):
        A += np.outer(
            original_points[match.original[i], :] - center_original, 
            detected_points[match.detected[i], :] - center_detected
        )
    U, S, V = np.linalg.svd(A)
    rotation = V @ U.T
    
    final_image = cv.resize(cv.imread(file_path), (1024, 1024))
    for i in range(m):
        x, y = original_points[i, :]
        x_, y_ = rotation @ (original_points[i, :] - center_original) + center_detected
        cv.circle(final_image, (int(x_), int(y_)), 5, (0, 0, 255), -1)
    
    if not args.no_plot:
        cv.imshow('final_image', cv.resize(final_image, (512, 512)))
        cv.waitKey(0)
        cv.destroyAllWindows()
    if args.save:
        cv.imwrite(os.path.join(args.outdir, file_name + '-final.png'), final_image)

x = list(range(1, len(candidatesInEachStep) + 1))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, candidatesInEachStep, 'g-')
ax2.plot(x, biggestCandidateInEachStep, 'b-')

if not args.no_plot:
    plt.show()
if args.save:
    fig.savefig(os.path.join(args.outdir, file_name + '-process.png'))