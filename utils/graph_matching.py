import numpy as np
import time
from collections import namedtuple

Match = namedtuple('Match', ['original', 'detected', 'size'])

def expandMatch(match, new, step, original_distance_matrix, detected_distance_matrix, error):
    for i, j in zip(match.detected, match.original):
        if np.linalg.norm(detected_distance_matrix[i, step] - original_distance_matrix[j, new]) > error:
            return False
    return True



def getDistanceMatrix(points):
    return np.array([np.linalg.norm(p - points, axis=1) for p in points])



def getInitialCandidates(detected_points, original_points, detected_distance_matrix, original_distance_matrix, error):
    n, m = len(detected_points), len(original_points)
    matches = []
    for i in range(m):
        for j in range(m):
            if abs(detected_distance_matrix[0, 1] - original_distance_matrix[i, j]) < error:
                matches.append(Match(original=(i, j), detected=(0, 1), size=2))
    return matches



def findMatch(detected_points, original_points, error, backup_iterations, initial_matches=None):
    detected_distance_matrix = getDistanceMatrix(detected_points)
    original_distance_matrix = getDistanceMatrix(original_points)
    n, m = len(detected_points), len(original_points)
    if initial_matches is None:
        matches = getInitialCandidates(detected_points, original_points, detected_distance_matrix, original_distance_matrix, error)
    else:
        matches = initial_matches
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
                if i not in match.original and expandMatch(match, i, step, original_distance_matrix, detected_distance_matrix, error):
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
    
    end_time = time.time()
    print()
    print(f'Number of matches: {len(matches)}')
    if len(matches) != 0:
        print(f'Biggest candidate: {biggestCandidateInEachStep[-1]}')


    bestMatches = [match for match in matches if match.size == biggestCandidateInEachStep[-1]]
    print(f'Number of best matches: {len(bestMatches)}')
    print(f'Time: {end_time - start_time}')

    return bestMatches, biggestCandidateInEachStep, candidatesInEachStep

