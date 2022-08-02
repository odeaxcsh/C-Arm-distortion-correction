from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

error1 = 25
error2 = 25


df_org = pd.read_csv('sample.csv')

original_x, original_y = df_org.iloc[:, 0], df_org.iloc[:, 1]
original_points = np.array([original_x, original_y]).T


detected_points = np.random.permutation(original_points)[:50] + np.random.uniform(-10, 10, (50, 2))

np.random.shuffle(detected_points)
np.random.shuffle(original_points)

original_points = original_points[:100, :]
original_x, original_y = original_points[:, 0], original_points[:, 1]

detected_x, detected_y = detected_points[:, 0], detected_points[:, 1]




print(f'''
    number of detected points: {len(detected_points)}
    number of original points: {len(original_points)}
    ratio of detected points: {len(detected_points) / len(original_points)}
''')

outliers = []
for i, p in enumerate(detected_points):
    dist = np.min(np.linalg.norm(p - original_points, axis=1))
    if dist > error1:
        print(f'Outlier detected at {i} with distance {dist}')
        outliers.append(i)

plt.xlim(0, max(original_x))
plt.ylim(0, max(original_y))
plt.plot(original_x, original_y, 'bo', label='original')
plt.plot(detected_x, detected_y, 'ro', label='detected')
plt.plot(detected_points[outliers, 0], detected_points[outliers, 1], 'gx', label='outliers')
plt.show()


def getDistanceMatrix(points):
    return np.array([np.linalg.norm(p - points, axis=1) for p in points])

original_distance_matrix = getDistanceMatrix(original_points)
detected_distance_matrix = getDistanceMatrix(detected_points)

n, m = len(detected_points), len(original_points)

Match = namedtuple('Match', ['original', 'detected', 'size'])


matches = []
for i in range(m):
    for j in range(m):
        if np.linalg.norm(detected_distance_matrix[0, 1] - original_distance_matrix[i, j]) < error1:
            matches.append(Match(original=(i, j), detected=(0, 1), size=2))
print(len(matches))


def isOkay(match, new, step, original_distance_matrix, detected_distance_matrix):
    for i, j in zip(match.detected, match.original):
        if np.linalg.norm(detected_distance_matrix[i, step] - original_distance_matrix[j, new]) > error2:
            return False
    return True


candidatesInEachStep = [len(matches)]

print('Number of matches in each step:')
print('\t 1. ', len(matches))

for step in range(2, len(detected_points)):
    new_matches = []
    for match in matches:
        found = False
        for i in range(m):
            if isOkay(match, i, step, original_distance_matrix, detected_distance_matrix):
                new_matches.append(Match(
                    original=(*match.original, i),
                    detected=(*match.detected, step),
                    size=match.size + 1
                ))
                found = True    
    if len(new_matches) > 0:
        matches = new_matches

    ###########################
    # containing_outlier = [match for match in matches if match.detected[-1] == step - 2]

    # matches = [match for match in matches if match.detected[-1] > step - 2]
    ###########################
    
    print(f'\t {step}. ', len(matches))
    candidatesInEachStep.append(len(matches))

    # draw_matches = matches
    # for match in draw_matches:
    #     plt.clf()

    #     for i in range(step + 1):
    #         plt.plot(detected_points[i, 0], detected_points[i, 1], 'rx')
    #     plt.plot(detected_points[step + 1, 0], detected_points[step + 1, 1], 'yx')


    #     for i in d[step + 1]:
    #         plt.plot(original_points[i, 0], original_points[i, 1], 'go')
        
    #     x = [original_points[i, 0] for i in match] + [original_points[match[0], 0]]
    #     y = [original_points[i, 1] for i in match] + [original_points[match[0], 1]]

    #     plt.plot(x, y, 'b-')

    #     plt.xlim(0, max(original_x))
    #     plt.ylim(0, max(original_y))
    #     plt.pause(1 / len(matches))
    # plt.pause(0.1)


print()
print(f'Number of matches: {len(matches)}')
if len(matches) != 0:
    print(f'Biggest candidate: {max(match.size for match in matches)}')

plt.plot(list(range(len(candidatesInEachStep))), candidatesInEachStep)
plt.show()
