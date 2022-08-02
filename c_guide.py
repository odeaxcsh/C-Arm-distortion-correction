import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_org = pd.read_csv('sample.csv')

original_x, original_y = df_org.iloc[:, 0], df_org.iloc[:, 1]
original_points = np.array([original_x, original_y]).T

error1 = 25
error2 = 25

detected_points = np.random.permutation(original_points)[:50] + np.random.uniform(-10, 10, (50, 2))


np.random.shuffle(detected_points)
np.random.shuffle(original_points)


# original_points = original_points[:100, :]
original_x, original_y = original_points[:, 0], original_points[:, 1]

detected_x, detected_y = detected_points[:, 0], detected_points[:, 1]


# Find nearest point of original_points for each point in detected_points
with open('C-Guide_correspondance.txt', 'w') as f: 
    for i, p in enumerate(detected_points):
        j = np.argmin(np.linalg.norm(p - original_points, axis=1))
        f.write(str(i) + ' ' + str(j) + '\n')



print(f'''
    number of detected points: {len(detected_points)}
    number of original points: {len(original_points)}
    ratio of detected points: {len(detected_points) / len(original_points)}
''')


# set plot X, Y limits
plt.xlim(0, max(original_x))
plt.ylim(0, max(original_y))
plt.plot(original_x, original_y, 'bo', label='original')
plt.plot(detected_x, detected_y, 'ro', label='detected')
plt.show()


def getDistanceMatrix(points):
    return np.array([np.linalg.norm(p - points, axis=1) for p in points])


original_distance_matrix = getDistanceMatrix(original_points)
detected_distance_matrix = getDistanceMatrix(detected_points)




d = {}
for i in range(len(detected_points)):
    d[i] = []
    for j in range(len(original_points)):
        d[i].append(j)
print(sum([len(v) for v in d.values()]))

matches = []
for i in d[0]:
    for j in d[1]:
        if np.linalg.norm(detected_distance_matrix[0, 1] - original_distance_matrix[i, j]) < error1:
            matches.append((i, j))
print(len(matches))


def isOkay(matches, new, step, original_distance_matrix, detected_distance_matrix):
    for i, j in enumerate(matches):
        if np.linalg.norm(detected_distance_matrix[i, step] - original_distance_matrix[j, new]) > error2:
            return False
    return True


candidates = [len(matches)]

for step in range(2, len(detected_points)):
    new_matches = []
    print(f'Start Step {step}')
    for match in matches:
        found = False
        for i in d[step]:
            if isOkay(match, i, step, original_distance_matrix, detected_distance_matrix):
                new_matches.append((*match, i))
                found = True
                # print(*match, i)
        
    matches = new_matches
    print(f'{step} {len(matches)}')
    candidates.append(len(matches))
    
    draw_matches = [match for match in matches if len(match) > step + 1 - int(step / 3)]
    for match in draw_matches:
        plt.clf()

        for i in range(step + 1):
            plt.plot(detected_points[i, 0], detected_points[i, 1], 'rx')
        plt.plot(detected_points[step + 1, 0], detected_points[step + 1, 1], 'yx')


        for i in d[step + 1]:
            plt.plot(original_points[i, 0], original_points[i, 1], 'go')
        
        x = [original_points[i, 0] for i in match] + [original_points[match[0], 0]]
        y = [original_points[i, 1] for i in match] + [original_points[match[0], 1]]

        plt.plot(x, y, 'b-')

        plt.xlim(0, max(original_x))
        plt.ylim(0, max(original_y))
        plt.pause(1 / len(matches))
    plt.pause(0.1)


print(len(match))
with open('C-Guide_correspondance-match.txt', 'w') as f:
    for i, m in enumerate(match):
        f.write(str(i) + ' ' + str(m) + '\n')
    
plt.plot(list(range(len(candidates))), candidates, )
plt.show()
