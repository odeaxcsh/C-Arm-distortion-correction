from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


###### Program Parameters ######
error1 = 25
error2 = 25
backup_iterations = 10
################################

###### Testnig Parameters ######
random_noise = 10

detected_sample_size = 50
original_sample_size = 60
random_points_number = 40
################################


df_org = pd.read_csv('sample.csv')

original_x, original_y = df_org.iloc[:, 0], df_org.iloc[:, 1]
original_points = np.array([original_x, original_y]).T

detected_points = np.random.permutation(original_points)[:detected_sample_size] + np.random.uniform(-random_noise, random_noise, (detected_sample_size, 2))

np.random.shuffle(detected_points)
np.random.shuffle(original_points)

random_points = np.random.uniform(min(original_x), max(original_x), (random_points_number, 2))
detected_points = np.vstack((detected_points, random_points))

a, b = 0, 0
maxDistance = np.linalg.norm(detected_points[0] - detected_points[0])
for i in range(len(detected_points)):
    for j in range(len(detected_points)):
        if np.linalg.norm(detected_points[i] - detected_points[j]) < maxDistance:
            a, b = i, j


detected_points[0], detected_points[a] = detected_points[a], detected_points[0]
detected_points[1], detected_points[b] = detected_points[b], detected_points[1]

original_points = original_points[:original_sample_size, :]
original_x, original_y = original_points[:, 0], original_points[:, 1]

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

outliers = []
for i, p in enumerate(detected_points):
    dist = np.min(np.linalg.norm(p - original_points, axis=1))
    if dist > error1:
        outliers.append(i)

if 1 in outliers or 0 in outliers:
    print('Warning: Initial guess contains outliers')
    input('Press enter to continue')

n_outliers = len(outliers)
print(f'''
    number of outliers: {n_outliers}
    ratio of outliers: {n_outliers / len(detected_points)}
''')

plt.xlim(0, max(original_x))
plt.ylim(0, max(original_y))
plt.plot(original_x, original_y, 'bo', label='original')
plt.plot(detected_x, detected_y, 'ro', label='detected')
plt.plot(detected_points[outliers, 0], detected_points[outliers, 1], 'gx', label='outliers')
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
    plt.xlim(0, max(original_x))
    plt.ylim(0, max(original_y))
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
