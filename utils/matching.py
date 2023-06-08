import numpy as np


def find_transformation(P, Q):
    P_c, Q_c = P.mean(axis=0), Q.mean(axis=0)
    P_t, Q_t = P - P_c, Q - Q_c
    W = np.dot(Q_t.T, P_t)
    U, _, V = np.linalg.svd(W)
    # if np.linalg.det(V) > 0:
    #     V *= -1
    R = np.dot(U, V)
    t = Q_c.T - np.dot(R, P_c.T)
    return R, t



def transform(P, R, t):
    return np.dot(R, P.T).T + t.T
