import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import ot
import ot.plot
from sklearn.datasets import make_moons
from skimage import transform
from pycpd import AffineRegistration

from math import log2, ceil
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.transformers import (
    TreeClassificationTransformer,
    NeuralClassificationTransformer,
)
from proglearn.voters import TreeClassificationVoter, KNNClassificationVoter
from proglearn.sims import generate_gaussian_parity

from joblib import Parallel, delayed

import seaborn as sns


def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

def nearest_neighbor(src, dst, y_src, y_dst, class_aware=True):
    assert src.shape == dst.shape

    distances = np.zeros(y_src.shape)
    indices = np.zeros(y_src.shape, dtype=int)

    if class_aware:
        class1_src = np.where(y_src == 1)[0]
        class0_src = np.where(y_src == 0)[0]
        class1_dst = np.where(y_dst == 1)[0]
        class0_dst = np.where(y_dst == 0)[0]

        neigh_1 = NearestNeighbors(n_neighbors=1)
        neigh_1.fit(dst[class1_dst])
        distances_1, indices_1 = neigh_1.kneighbors(
            src[class1_src], return_distance=True
        )

        neigh_2 = NearestNeighbors(n_neighbors=1)
        neigh_2.fit(dst[class0_dst])
        distances_2, indices_2 = neigh_2.kneighbors(
            src[class0_src], return_distance=True
        )

        closest_class1 = class1_src[indices_1]
        closest_class0 = class0_src[indices_2]

        count = 0
        for i in class1_src:
            distances[i] = distances_1[count]
            indices[i] = closest_class1[count]
            count = count + 1

        count = 0
        for i in class0_src:
            distances[i] = distances_2[count]
            indices[i] = closest_class0[count]
            count = count + 1

    else:
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)

    return distances.ravel(), indices.ravel()


def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def icp(A, B, y_src, y_dst, init_pose=None, max_iterations=500, tolerance=1e-26):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    imbalance = []

    class1_src = np.where(y_src == 1)[0]
    class0_src = np.where(y_src == 0)[0]
    class1_dst = np.where(y_dst == 1)[0]
    class0_dst = np.where(y_dst == 0)[0]

    imbalance.append(len(class1_src))
    imbalance.append(len(class0_src))
    imbalance.append(len(class1_dst))
    imbalance.append(len(class0_dst))

    mi = min(imbalance)

    X_1 = src[:, class1_src[0:mi]]
    X_2 = src[:, class0_src[0:mi]]

    src_subsample = np.concatenate((X_1, X_2), 1)
    y_src_sub = np.concatenate((np.ones(mi), np.zeros(mi)))

    X_1 = dst[:, class1_dst[0:mi]]
    X_2 = dst[:, class0_dst[0:mi]]
    dst_subsample = np.concatenate((X_1, X_2), 1)
    y_dst_sub = np.concatenate((np.ones(mi), np.zeros(mi)))

    for i in range(max_iterations):

        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(
            src_subsample[:m, :].T, dst_subsample[:m, :].T, y_src_sub, y_dst_sub
        )
        # distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T, y_src, y_dst)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(
            src_subsample[:m, :].T, dst_subsample[:m, indices].T
        )
        # T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src_subsample = np.dot(T, src_subsample)
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    # T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, src, i


def cpd_reg(template, target, max_iter=50):    
    registration = AffineRegistration(X=target, Y=template, max_iterations=max_iter)
    deformed_template = registration.register(template)
    
    return deformed_template[0]

def plot_xor_rxor(data, labels, title):
    colors = sns.color_palette("Dark2", n_colors=2)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(data[:, 0], data[:, 1], c=get_colors(colors, labels), s=50)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=30)
    # plt.tight_layout()
    ax.axis("off")
    plt.show()
    

def experiment(
    n_task1,
    n_task2,
    n_test=0.4,
    task1_angle=0,
    task2_angle=np.pi / 2,
    n_trees=10,
    max_depth=None,
    random_state=None,
    register_cpd=False,
    register_otp=False,
    register_icp=False,
):

    if n_task1 == 0 and n_task2 == 0:
        raise ValueError("Wake up and provide samples to train!!!")

    if random_state != None:
        np.random.seed(random_state)

    errors = np.zeros(6, dtype=float)

    default_transformer_class = TreeClassificationTransformer
    default_transformer_kwargs = {"kwargs": {"max_depth": max_depth}}

    default_voter_class = TreeClassificationVoter
    default_voter_kwargs = {}

    default_decider_class = SimpleArgmaxAverage
    default_decider_kwargs = {"classes": np.arange(2)}
    progressive_learner = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )
    uf = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )
    naive_uf = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
        default_decider_kwargs=default_decider_kwargs,
    )

    # source data
    X_task1, y_task1 = sample_cc18(n_task1, angle_params=task1_angle)
    test_task1, test_label_task1 = sample_cc18(
        n_test, angle_params=task1_angle
    )
    
    # target data
    tform1 = transform.AffineTransform(shear=0*np.pi/180)
    tform2 = transform.AffineTransform(shear=task2_angle)
    
    X_task2, y_task2 = sample_cc18(n_task2, angle_params=task1_angle)
    test_task2, test_label_task2 = sample_cc18(
        n_test, angle_params=task1_angle
    )
    
    # Transform training set
    X1_top = X_task2[X_task2[:,1] >= 0]
    X1_bottom = X_task2[X_task2[:,1] < 0]
    y1_top = y_task2[X_task2[:,1] >= 0]
    y1_bottom = y_task2[X_task2[:,1] < 0]

    m = X1_top.shape[1]
    src = np.ones((m + 1, X1_top.shape[0]))
    src[:m, :] = np.copy(X1_top.T)

    src = np.dot(tform1.params, src)
    X2_top = src.T[:, 0:2]
    y2_top = y1_top

    m = X1_bottom.shape[1]
    src = np.ones((m + 1, X1_bottom.shape[0]))
    src[:m, :] = np.copy(X1_bottom.T)

    src = np.dot(tform2.params, src)
    X2_bottom = src.T[:, 0:2]
    y2_bottom = y1_bottom

    X_task2 = np.concatenate((X2_top, X2_bottom))
    y_task2 = np.concatenate((y2_top, y2_bottom)) 
 
    if register_cpd:
        
        X_task2 = cpd_reg(X_task2.copy(), X_task1.copy())

    if register_otp:
        ot_sinkhorn = ot.da.SinkhornTransport(reg_e=1e-2)
        ot_sinkhorn.fit(Xs=X_task2.copy(), Xt=X_task1.copy(), ys=y_task2.copy(), yt=y_task1.copy())
        transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=X_task2.copy())
        X_task2 = transp_Xs_sinkhorn
    
    if register_icp:
        T, X_3, i = icp(X_task2.copy(), X_task1.copy(), y_task2.copy(), y_task1.copy())
        X_task2 = X_3.T[:, 0:2]

    progressive_learner.add_task(X_task1, y_task1, num_transformers=n_trees)
    progressive_learner.add_task(X_task2, y_task2, num_transformers=n_trees)

    uf.add_task(X_task1, y_task1, num_transformers=2 * n_trees)
    uf.add_task(X_task2, y_task2, num_transformers=2 * n_trees)

    uf_task1 = uf.predict(test_task1, transformer_ids=[0], task_id=0)
    l2f_task1 = progressive_learner.predict(test_task1, task_id=0)

    errors[0] = 1 - np.mean(uf_task1 == test_label_task1)
    errors[1] = 1 - np.mean(l2f_task1 == test_label_task1)

    return errors


def bte_v_angle(angle_sweep, task1_sample, task2_sample, mc_rep, register_cpd=False, register_otp=False, register_icp=False):
    mean_te = np.zeros(len(angle_sweep), dtype=float)
    for ii, angle in enumerate(angle_sweep):
        error = np.array(
            Parallel(n_jobs=-1, verbose=1)(
                delayed(experiment)(
                    task1_sample,
                    task2_sample,
                    task2_angle=angle * np.pi / 180,
                    max_depth=ceil(log2(task1_sample)),
                    register_cpd=register_cpd,
                    register_otp=register_otp,
                    register_icp=register_icp
                )
                for _ in range(mc_rep)
            )
        )

        mean_te[ii] = np.mean(error[:, 0]) / np.mean(error[:, 1])

    return mean_te


def plot_bte_v_angle(angle_sweep, mean_te1, mean_te2, mean_te3, mean_te4, mean_te5):
    fontsize=50
    labelsize = 36
    
    sns.set_context("talk")
    fig = plt.figure(constrained_layout=True, figsize=(25, 23))
    gs = fig.add_gridspec(6, 6)
    ax = fig.add_subplot(gs[:6, :6])
    task = ["No adaptation", "ICP", "CPD", "OT-exact", "OT-IT"]
    ax.plot(angle_sweep, mean_te1, c="r", linewidth=3, label=task[0])
    ax.plot(angle_sweep, mean_te2, c="b", linewidth=3, label=task[1])
    ax.plot(angle_sweep, mean_te3, c="g", linewidth=3, label=task[2])
    ax.plot(angle_sweep, mean_te4, c="purple", linewidth=3, label=task[3])
    ax.plot(angle_sweep, mean_te5, c="orange", linewidth=3, label=task[4])
    ax.set_xticks(range(0, 91, 15))
    ax.tick_params(labelsize=labelsize)
    ax.set_xlabel("Angle of Rotation (Degrees)", fontsize=fontsize)
    ax.set_ylabel("Backward Transfer Efficiency (XOR)", fontsize=fontsize)
    ax.set_yticks([0.9, 1, 1.1, 1.2, 1.3])
    # ax.set_title("XOR vs. Rotated-XOR", fontsize = fontsize)
    ax.hlines(1, 0, 90, colors="grey", linestyles="dashed", linewidth=1.5)
    ax.legend(loc="lower left", fontsize=40, frameon=False)

    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top_side = ax.spines["top"]
    top_side.set_visible(False)

