import numpy as np


def normalize(P, s=1, t=512):
    return (P - t) / s


def normalize_i(P, s=1, t=512):
    return (P * s) + t


def barrel_estimation(P, Q):
    P, Q = normalize(P), normalize(Q)

    r1 = np.abs(P[:, 0])
    A1 = np.vstack((r1**2, r1**4)).T
    b1 = (Q[:, 0] / P[:, 0] - 1)
    x, _, _, _ = np.linalg.lstsq(A1, b1, rcond=None)


    r2 = np.abs(P[:, 1])
    A2 = np.vstack((r2**2, r2**4)).T
    b2 = Q[:, 1] / P[:, 1] - 1
    y, _, _, _ = np.linalg.lstsq(A2, b2, rcond=None)

    return np.hstack((x, y))



def transform(P, k):
    P = normalize(P)
    r1, r2 = np.abs(P[:, 0]), np.abs(P[:, 1])
    T = np.vstack((
        P[:, 0] * (1 + k[0] * r1**2 + k[1] * r1**4),
        P[:, 1] * (1 + k[2] * r1**2 + k[3] * r1**4)
    )).T
    return normalize_i(T)



def transform_inverse(Q, k):
    Q = normalize(Q)
    r1, r2 = np.abs(Q[:, 0]), np.abs(Q[:, 1])
    T = np.vstack((
        Q[:, 0] / (1 + k[0] * r1**2 + k[1] * r1**4),
        Q[:, 1] / (1 + k[2] * r2**2 + k[3] * r2**4)
    )).T
    return normalize_i(T)



def max_distortion(P, k):
    transformed = transform(P, k)
    return np.max(np.linalg.norm(transformed - P, axis=1))

