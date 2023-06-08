from utils import matching, detect, distortion, graph_matching

import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import os


error = 30
backup_iterations = 0

file_path = 'org/58.tif'
file_name, file_extension = os.path.splitext(file_path.split('/')[-1])
df_org = pd.read_csv('reference.csv', index_col=0)
original_x, original_y = df_org.iloc[:, 0], df_org.iloc[:, 1]
original_x, original_y = np.array(original_x), np.array(original_y)
original_points = np.array([original_x, original_y]).T


image = cv.imread(file_path)
image = detect.preprocess(image)
detected_points = detect.detect(image)
detected_x, detected_y = detected_points[:, 0], detected_points[:, 1]


n, m = len(detected_points), len(original_points)
bestMatches, biggestCandidateInEachStep, candidatesInEachStep = graph_matching.findMatch(
    detected_points, original_points,
    error, 
    backup_iterations
)

if len(bestMatches) == 0:
    print('No matches found')
    exit(-1)
elif len(bestMatches) > 1:
    print('Warning: multiple matches found')
    input('Press any key to continue...')


match = bestMatches[0]

P = original_points[match.original, :]
Q = detected_points[match.detected, :]

R, t = matching.find_transformation(P, Q)

transformed = matching.transform(original_points, R, t)

P = matching.transform(P, R, t)

k = distortion.barrel_estimation(P, Q)
t = distortion.transform(P, k)
z = distortion.max_distortion(P, k)


outliers = [i for i in range(n) if i not in match.detected]
inliers = [i for i in range(n) if i in match.detected]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')
ax.set_xlim(0, max(original_x))
ax.set_ylim(0, max(original_x))
ax.scatter(transformed[:, 0], transformed[:, 1], s=12, color='b', label='phantom')
ax.scatter(detected_x, detected_y, s=15, edgecolors='g', label='detected', facecolors='none')
ax.scatter(detected_points[outliers, 0], detected_points[outliers, 1], s=30, edgecolors='r', facecolors='none', label='not matched')
ax.scatter(t[:, 0], t[:, 1], s=12, color='r', label='Distortion Model')

for i, j in zip(match.detected, match.original):
    ax.scatter(detected_points[i, 0], detected_points[i, 1], s=12, color='g')
    ax.plot([detected_points[i, 0], transformed[j, 0]], [detected_points[i, 1], transformed[j, 1]], 'g-', linewidth=1)

plt.legend(loc='best')
plt.axis('off')
plt.show()
