import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
import glob
import os

import seaborn as sns

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])



def make_results_video(opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, seed, proj_matrix=None):
    if proj_matrix is not None:
        unproj_matrix = np.linalg.pinv(proj_matrix)
        # a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
        generated_samples = generated_samples @ unproj_matrix
        img_shape = (1, 28, 28)
        generated_samples = np.reshape(generated_samples, (generated_samples.shape[0], *img_shape))
        generated_samples = torch.from_numpy(generated_samples)
    else:
        generated_samples = torch.from_numpy(generated_samples)

    for i in range(len(res_student)-1):

        generated_sample = generated_samples[i].squeeze()
        generated_label = generated_labels[i]

        if generated_label == 0.0:
            generated_label = opt.class_1
        else:
            generated_label = opt.class_2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(20, 5.8)
        #ax1.plot(a, b, '-k', label='Teacher Classifier')
        # ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
        ax1.imshow(generated_sample, cmap='gray')
        # ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1], marker='x')
        ax1.set_title("Data Generation - Label {}".format((generated_label)))
        # ax1.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax1.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax1.legend(loc="upper right")

        #ax2.plot(a, b, '-k', label='Teacher Classifier')
        # ax2.plot(a_baseline[i], b_baseline[i], '-g', label='IMT Classifier')
        # ax2.scatter(X[:, 0], X[:, 1], c=Y)
        # ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1, 0], marker='x')
        # ax2.set_title("Data Selection (IMT)")
        # ax2.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax2.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax2.legend(loc="upper right")

        ax2.plot(res_sgd[:i+1], c='g', label="SGD %s" % opt.data_mode)
        ax2.plot(res_baseline[:i+1], c='b', label="IMT %s" % opt.data_mode)
        ax2.plot(res_student[:i+1], c='r', label="Student %s" % opt.data_mode)
        # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")

        ax3.plot(w_diff_sgd[:i+1], 'g', label="SGD %s" % opt.data_mode)
        ax3.plot(w_diff_baseline[:i+1], 'b', label="IMT %s" % opt.data_mode)
        ax3.plot(w_diff_student[:i+1], 'r', label="Student %s" % opt.data_mode)
        ax3.legend(loc="lower left")
        ax3.set_title("w diff " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Distance between $w^t$ and $w^*$")
        #ax3.set_aspect('equal')

        video_dir = os.path.join(opt.log_path, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        plt.savefig(video_dir + "/file%03d.png" % i)

        plt.close()

    # os.chdir(video_dir)
    images = []
    for file_name in sorted(glob.glob(video_dir + "/*.png")):
        # print(file_name)
        images.append(imageio.imread(file_name))
        # os.remove(file_name)
    gif_path = os.path.join(video_dir, 'results_{}_{}_{}.gif'.format(opt.data_mode, epoch, seed))
    imageio.mimsave(gif_path, images, fps=20)
    # optimize(gif_path)

    '''
    os.chdir(CONF.PATH.OUTPUT)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    '''


def make_results_img(opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, seed, proj_matrix=None):

    print("generated samples", generated_samples.shape)

    if proj_matrix is not None:
        unproj_matrix = np.linalg.pinv(proj_matrix)
        generated_samples = generated_samples @ unproj_matrix
        img_shape = (1, 28, 28)
        generated_samples = np.reshape(generated_samples, (generated_samples.shape[0], *img_shape))
        generated_samples = torch.from_numpy(generated_samples)
    else:
        generated_samples = torch.from_numpy(generated_samples)

    generated_sample = generated_samples[-1].squeeze()
    generated_label = generated_labels[-1]

    if generated_label == 0.0:
        generated_label = opt.class_1
    else:
        generated_label = opt.class_2

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 5.8)
    # ax1.plot(a_student[-1], b_student[-1], '-r', label='Optimizer Classifier')
    # ax1.scatter(X[:, 0], X[:, 1], c=Y)
    # ax1.scatter(generated_samples[:, 0], generated_samples[:, 1], c=generated_labels[:], marker='x')
    ax1.imshow(generated_sample, cmap='gray')
    ax1.legend(loc="upper right")
    ax1.set_title("Data Generation - Label {}".format((generated_label)))
    #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
    #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

    ax2.plot(res_sgd, c='g', label="SGD %s" % opt.data_mode)
    ax2.plot(res_baseline, c='b', label="IMT %s" % opt.data_mode)
    ax2.plot(res_student, c='r', label="Student %s" % opt.data_mode)
    # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")

    ax3.plot(w_diff_sgd, 'g', label="SGD %s" % opt.data_mode)
    ax3.plot(w_diff_baseline, 'b', label="IMT %s" % opt.data_mode)
    ax3.plot(w_diff_student, 'r', label="Student %s" % opt.data_mode)
    ax3.legend(loc="lower left")
    ax3.set_title("w diff " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Distance between $w^t$ and $w^*$")
    #ax3.set_aspect('equal')

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_path = os.path.join(save_folder, 'results_{}_{}_{}.png'.format(opt.data_mode, epoch, seed))
    plt.savefig(img_path)
    plt.close()

    '''
    n_rows = 10
    indices = torch.randint(0, len(generated_samples), (n_rows**2,))
    labels = generated_labels[indices]
    samples = generated_samples[indices]

    # gen_imgs = samples @ unproj_matrix

    img_shape = (1, 28, 28)
    # gen_imgs = samples
    im = np.reshape(samples, (samples.shape[0], *img_shape))
    im = torch.from_numpy(im)

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    grid = make_grid(im, nrow=10, normalize=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
    ax.axis('off')
    plt.title("Fake Images, Label", )
    img_path = os.path.join(save_folder, "results_{}_imgs.png".format(epoch))
    plt.savefig(img_path)
    plt.close()

    # plt.figure(figsize=(10, 10)) # specifying the overall grid size

    # for i in range(25):
    #    plt.subplot(5, 5, i+1)    # the number of images in the grid is 5*5 (25)
    #    plt.imshow(im[:, :, i], cmap="gray")

    # plt.axis("off")

    plt.figure(figsize=(10, 10))
    # plt.plot(res_example, 'go', label="linear classifier", alpha=0.5)
    # plt.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % opt.teaching_mode, alpha=0.5)
    # plt.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % opt.teaching_mode, alpha=0.5)
    plt.plot(w_diff_example, 'go', label="linear classifier", alpha=0.5)
    plt.plot(w_diff_baseline, 'bo', label="%s & baseline" % opt.teaching_mode, alpha=0.5)
    plt.plot(w_diff_student, 'ro', label="%s & linear classifier" % opt.teaching_mode, alpha=0.5)
    # plt.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    plt.legend(loc="upper right")
    plt.title("Test Set Accuracy")
    #plt.set_aspect('equal')

    img_path = os.path.join(save_folder, "results_{}_w_diff.png".format(epoch))
    plt.savefig(img_path)
    plt.close()
    '''

def make_results(opt, res_sgd, res_baseline, res_student, res_student_label, res_label, res_imt_label, w_diff_sgd, w_diff_baseline, w_diff_student, w_diff_student_label, w_diff_label, w_diff_imt_label, epoch, seed):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13, 5.8)

    ax1.plot(res_sgd, c='g', label="SGD %s" % opt.data_mode)
    ax1.plot(res_baseline, c='b', label="IMT %s" % opt.data_mode)
    ax1.plot(res_student, c='r', label="Student %s" % opt.data_mode)
    ax1.plot(res_label, c='c', label="SGD + Label %s" % opt.data_mode)
    ax1.plot(res_imt_label, c='m', label="IMT + Label %s" % opt.data_mode)
    ax1.plot(res_student_label, c='k', label="Student + Label %s" % opt.data_mode)
    # ax1.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    ax1.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left")

    ax2.plot(w_diff_sgd, 'g', label="SGD %s" % opt.data_mode)
    ax2.plot(w_diff_baseline, 'b', label="IMT %s" % opt.data_mode)
    ax2.plot(w_diff_student, 'r', label="Student %s" % opt.data_mode)
    ax2.plot(w_diff_label, c='c', label="SGD + Label %s" % opt.data_mode)
    ax2.plot(w_diff_imt_label, c='m', label="IMT + Label %s" % opt.data_mode)
    ax2.plot(w_diff_student_label, c='k', label="Student + Label %s" % opt.data_mode)
    ax2.legend(loc="lower left")
    ax2.set_title("w diff " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Distance between $w^t$ and $w^*$")
    #ax2.set_aspect('equal')

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_path = os.path.join(save_folder, 'results_{}_{}_{}_tmp.png'.format(opt.data_mode, epoch, seed))
    plt.savefig(img_path)
    plt.close()


def make_results_img_2d(opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, seed, res_student_label=None, w_diff_student_label=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 5.8)
    # ax1.plot(a_student[-1], b_student[-1], '-r', label='Optimizer Classifier')
    ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, edgecolors='k')
    ax1.scatter(generated_samples[:, 0], generated_samples[:, 1], c=generated_labels[:, 0], cmap=cm_bright, marker='^')
    ax1.legend(loc="upper right")
    ax1.set_title("Data Generation")
    #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
    #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

    ax2.plot(res_sgd, c='g', label="SGD %s" % opt.data_mode)
    ax2.plot(res_baseline, c='b', label="IMT %s" % opt.data_mode)
    ax2.plot(res_student, c='r', label="Student %s" % opt.data_mode)
    if res_student_label is not None:
        ax2.plot(res_student, c='k', label="Student + Label %s" % opt.data_mode)
    # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")

    ax3.plot(w_diff_sgd, 'g', label="SGD %s" % opt.data_mode)
    ax3.plot(w_diff_baseline, 'b', label="IMT %s" % opt.data_mode)
    ax3.plot(w_diff_student, 'r', label="Student %s" % opt.data_mode)
    if w_diff_student_label is not None:
        ax3.plot(w_diff_student_label, c='k', label="Student + Label %s" % opt.data_mode)
    ax3.legend(loc="lower left")
    ax3.set_title("w diff " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Distance between $w^t$ and $w^*$")
    #ax3.set_aspect('equal')

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_path = os.path.join(save_folder, 'results_{}_{}_{}.png'.format(opt.data_mode, epoch, seed))
    plt.savefig(img_path)
    plt.close()

def make_results_video_2d(opt, X, Y, generated_samples, generated_labels, res_sgd, res_baseline, res_student, w_diff_sgd, w_diff_baseline, w_diff_student, epoch, seed):
    # a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
    for i in range(len(res_student)):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_size_inches(20, 5.8)
        #ax1.plot(a, b, '-k', label='Teacher Classifier')
        # ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
        ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, edgecolors='k')
        ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1, 0], cmap=cm_bright, marker='^')
        ax1.set_title("Data Generation (Ours)")
        # ax1.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax1.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax1.legend(loc="upper right")

        #ax2.plot(a, b, '-k', label='Teacher Classifier')
        # ax2.plot(a_baseline[i], b_baseline[i], '-g', label='IMT Classifier')
        # ax2.scatter(X[:, 0], X[:, 1], c=Y)
        # ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1, 0], marker='x')
        # ax2.set_title("Data Selection (IMT)")
        # ax2.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax2.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax2.legend(loc="upper right")

        ax2.plot(res_sgd[:i+1], c='g', label="SGD %s" % opt.data_mode)
        ax2.plot(res_baseline[:i+1], c='b', label="IMT %s" % opt.data_mode)
        ax2.plot(res_student[:i+1], c='r', label="Student %s" % opt.data_mode)
        # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")

        ax3.plot(w_diff_sgd[:i+1], 'g', label="SGD %s" % opt.data_mode)
        ax3.plot(w_diff_baseline[:i+1], 'b', label="IMT %s" % opt.data_mode)
        ax3.plot(w_diff_student[:i+1], 'r', label="Student %s" % opt.data_mode)
        ax3.legend(loc="lower left")
        ax3.set_title("w diff " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Distance between $w^t$ and $w^*$")
        #ax3.set_aspect('equal')

        video_dir = os.path.join(opt.log_path, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        plt.savefig(video_dir + "/file%03d.png" % i)

        plt.close()

    # os.chdir(video_dir)
    images = []
    for file_name in sorted(glob.glob(video_dir + "/*.png")):
        # print(file_name)
        images.append(imageio.imread(file_name))
        # os.remove(file_name)
    gif_path = os.path.join(video_dir, 'results_{}_{}_{}.gif'.format(opt.data_mode, epoch, seed))
    imageio.mimsave(gif_path, images, fps=20)
    # optimize(gif_path)

    '''
    os.chdir(CONF.PATH.OUTPUT)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    '''


def plot_generated_samples_2d(opt, X, Y, a_star, b_star, a_student, b_student, generated_samples, generated_labels, epoch, seed):
    sns.set()

    save_folder = os.path.join(opt.log_path, "generated_samples")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    iterations = [1, 75, 150, 225, -1]
    for i in iterations:

        fig = plt.figure()
        fig.set_size_inches(5.8, 5.8)
        plt.plot(a_star, b_star, '-k', label='W star')
        plt.plot(a_student[i], b_student[i], '-r', label='Learned Classifier')
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, edgecolors='k')
        plt.scatter(generated_samples[:i, 0], generated_samples[:i, 1], c=generated_labels[:i, 0], cmap=cm_bright, marker='^')
        plt.legend(loc="upper right", fontsize=16)
        plt.xlim([X[:, 0].min()-0.4, X[:, 0].max()+0.4])
        plt.ylim([X[:, 1].min()-0.4, X[:, 1].max()+0.4])

        img_path = os.path.join(save_folder, 'paper_generated_samples_{}_{}_{}_{}.jpg'.format(opt.data_mode, epoch, seed, i))
        plt.savefig(img_path)
        plt.close()


def plot_generated_samples(opt, X, Y, generated_samples, generated_labels, epoch, seed):

    save_folder = os.path.join(opt.log_path, "generated_samples")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    iterations = [1, 75, 150, 225, -1]
    for i in iterations:

        generated_sample = generated_samples[i, :].squeeze()
        generated_label = generated_labels[i]

        if generated_label == 0.0:
            generated_label = opt.class_1
        else:
            generated_label = opt.class_2

        fig = plt.figure()
        fig.set_size_inches(5.8, 5.8)
        plt.imshow(generated_sample, cmap='gray')
        plt.legend(loc="upper right", fontsize=16)
        plt.title("Data Generation - Label {}".format(generated_label), fontsize=16)
        img_path = os.path.join(save_folder, 'paper_generated_samples_{}_{}_{}_{}.jpg'.format(opt.data_mode, epoch, seed, i))
        plt.savefig(img_path)
        plt.close()

def make_results_video_blackbox(opt, X, Y, generated_samples, generated_labels, res_sgd, res_student, w_diff_sgd, w_diff_student, epoch, seed, proj_matrix=None):
    if proj_matrix is not None:
        unproj_matrix = np.linalg.pinv(proj_matrix)
        # a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
        generated_samples = generated_samples @ unproj_matrix
        img_shape = (1, 28, 28)
        generated_samples = np.reshape(generated_samples, (generated_samples.shape[0], *img_shape))
        generated_samples = torch.from_numpy(generated_samples)
    else:
        generated_samples = torch.from_numpy(generated_samples)

    for i in range(len(res_student)-1):

        generated_sample = generated_samples[i].squeeze()
        generated_label = generated_labels[i]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(13.3, 5.8)
        #ax1.plot(a, b, '-k', label='Teacher Classifier')
        # ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
        ax1.imshow(generated_sample, cmap='gray')
        # ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1], marker='x')
        ax1.set_title("Data Generation (Ours)")
        # ax1.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax1.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax1.legend(loc="upper right")

        #ax2.plot(a, b, '-k', label='Teacher Classifier')
        # ax2.plot(a_baseline[i], b_baseline[i], '-g', label='IMT Classifier')
        # ax2.scatter(X[:, 0], X[:, 1], c=Y)
        # ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1, 0], marker='x')
        # ax2.set_title("Data Selection (IMT)")
        # ax2.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax2.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax2.legend(loc="upper right")

        ax2.plot(res_sgd[:i+1], c='g', label="SGD %s" % opt.data_mode)
        ax2.plot(res_student[:i+1], c='r', label="Student %s" % opt.data_mode)
        # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")


        video_dir = os.path.join(opt.log_path, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        plt.savefig(video_dir + "/file%03d.png" % i)

        plt.close()

    # os.chdir(video_dir)
    images = []
    for file_name in sorted(glob.glob(video_dir + "/*.png")):
        # print(file_name)
        images.append(imageio.imread(file_name))
        # os.remove(file_name)
    gif_path = os.path.join(video_dir, 'results_{}_{}_{}.gif'.format(opt.data_mode, epoch, seed))
    imageio.mimsave(gif_path, images, fps=20)
    # optimize(gif_path)

    '''
    os.chdir(CONF.PATH.OUTPUT)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    '''


def make_results_img_blackbox(opt, X, Y, generated_samples, generated_labels, res_sgd, res_student, w_diff_sgd, w_diff_student, epoch, seed, proj_matrix=None):

    print("generated samples", generated_samples.shape)

    if proj_matrix is not None:
        unproj_matrix = np.linalg.pinv(proj_matrix)
        # a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
        generated_samples = generated_samples @ unproj_matrix
        img_shape = (1, 28, 28)
        generated_samples = np.reshape(generated_samples, (generated_samples.shape[0], *img_shape))
        generated_samples = torch.from_numpy(generated_samples)
    else:
        generated_samples = torch.from_numpy(generated_samples)

    generated_sample = generated_samples[-1].squeeze()
    generated_label = generated_labels[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13.3, 5.8)
    # ax1.plot(a_student[-1], b_student[-1], '-r', label='Optimizer Classifier')
    # ax1.scatter(X[:, 0], X[:, 1], c=Y)
    # ax1.scatter(generated_samples[:, 0], generated_samples[:, 1], c=generated_labels[:], marker='x')
    ax1.imshow(generated_sample, cmap='gray')
    ax1.legend(loc="upper right")
    ax1.set_title("Data Generation (Ours)")
    #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
    #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

    ax2.plot(res_sgd, c='g', label="SGD %s" % opt.data_mode)
    ax2.plot(res_student, c='r', label="Student %s" % opt.data_mode)
    # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_path = os.path.join(save_folder, 'results_{}_{}_{}.png'.format(opt.data_mode, epoch, seed))
    plt.savefig(img_path)
    plt.close()

    '''
    n_rows = 10
    indices = torch.randint(0, len(generated_samples), (n_rows**2,))
    labels = generated_labels[indices]
    samples = generated_samples[indices]

    # gen_imgs = samples @ unproj_matrix

    img_shape = (1, 28, 28)
    # gen_imgs = samples
    im = np.reshape(samples, (samples.shape[0], *img_shape))
    im = torch.from_numpy(im)

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    grid = make_grid(im, nrow=10, normalize=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
    ax.axis('off')
    plt.title("Fake Images, Label", )
    img_path = os.path.join(save_folder, "results_{}_imgs.png".format(epoch))
    plt.savefig(img_path)
    plt.close()

    # plt.figure(figsize=(10, 10)) # specifying the overall grid size

    # for i in range(25):
    #    plt.subplot(5, 5, i+1)    # the number of images in the grid is 5*5 (25)
    #    plt.imshow(im[:, :, i], cmap="gray")

    # plt.axis("off")

    plt.figure(figsize=(10, 10))
    # plt.plot(res_example, 'go', label="linear classifier", alpha=0.5)
    # plt.plot(res_baseline[:i+1], 'bo', label="%s & baseline" % opt.teaching_mode, alpha=0.5)
    # plt.plot(res_student[:i+1], 'ro', label="%s & linear classifier" % opt.teaching_mode, alpha=0.5)
    plt.plot(w_diff_example, 'go', label="linear classifier", alpha=0.5)
    plt.plot(w_diff_baseline, 'bo', label="%s & baseline" % opt.teaching_mode, alpha=0.5)
    plt.plot(w_diff_student, 'ro', label="%s & linear classifier" % opt.teaching_mode, alpha=0.5)
    # plt.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    plt.legend(loc="upper right")
    plt.title("Test Set Accuracy")
    #plt.set_aspect('equal')

    img_path = os.path.join(save_folder, "results_{}_w_diff.png".format(epoch))
    plt.savefig(img_path)
    plt.close()
    '''

def make_results_img_2d_blackbox(opt, X, Y, generated_samples, generated_labels, res_sgd, res_student, w_diff_sgd, w_diff_student, epoch, seed):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(13.3, 5.8)
    # ax1.plot(a_student[-1], b_student[-1], '-r', label='Optimizer Classifier')
    ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, edgecolors='k')
    ax1.scatter(generated_samples[:, 0], generated_samples[:, 1], c=generated_labels[:, 0], cmap=cm_bright, marker='^')
    ax1.legend(loc="upper right")
    ax1.set_title("Data Generation")
    #ax1.set_xlim([X.min()-0.5, X.max()+0.5])
    #ax1.set_ylim([X.min()-0.5, X.max()+0.5])

    ax2.plot(res_sgd, c='g', label="SGD %s" % opt.data_mode)
    ax2.plot(res_student, c='r', label="Student %s" % opt.data_mode)
    # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
    ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")

    save_folder = os.path.join(opt.log_path, "imgs")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_path = os.path.join(save_folder, 'results_{}_{}_{}.png'.format(opt.data_mode, epoch, seed))
    plt.savefig(img_path)
    plt.close()


def make_results_video_2d_blackbox(opt, X, Y, generated_samples, generated_labels, res_sgd, res_student, w_diff_sgd, w_diff_student, epoch, seed):
    # a, b = plot_classifier(teacher, X.max(axis=0), X.min(axis=0))
    for i in range(len(res_student)):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(13.3, 5.8)
        #ax1.plot(a, b, '-k', label='Teacher Classifier')
        # ax1.plot(a_student[i], b_student[i], '-r', label='Optimizer Classifier')
        ax1.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright, edgecolors='k')
        ax1.scatter(generated_samples[:i+1, 0], generated_samples[:i+1, 1], c=generated_labels[:i+1, 0], cmap=cm_bright, marker='^')
        ax1.set_title("Data Generation (Ours)")
        # ax1.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax1.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax1.legend(loc="upper right")

        #ax2.plot(a, b, '-k', label='Teacher Classifier')
        # ax2.plot(a_baseline[i], b_baseline[i], '-g', label='IMT Classifier')
        # ax2.scatter(X[:, 0], X[:, 1], c=Y)
        # ax2.scatter(selected_samples[:i+1, 0], selected_samples[:i+1, 1], c=selected_labels[:i+1, 0], marker='x')
        # ax2.set_title("Data Selection (IMT)")
        # ax2.set_xlim([X.min()-0.5, X.max()+0.5])
        # ax2.set_ylim([X.min()-0.5, X.max()+0.5])
        # ax2.legend(loc="upper right")

        ax2.plot(res_sgd[:i+1], c='g', label="SGD %s" % opt.data_mode)
        ax2.plot(res_student[:i+1], c='r', label="Student %s" % opt.data_mode)
        # ax2.axhline(y=teacher_acc, color='k', linestyle='-', label="teacher accuracy")
        ax2.set_title("Test accuracy " + str(opt.data_mode) + " (class : " + str(opt.class_1) + ", " + str(opt.class_2) + ")")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.legend(loc="lower right")

        video_dir = os.path.join(opt.log_path, "video")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        plt.savefig(video_dir + "/file%03d.png" % i)

        plt.close()

    # os.chdir(video_dir)
    images = []
    for file_name in sorted(glob.glob(video_dir + "/*.png")):
        # print(file_name)
        images.append(imageio.imread(file_name))
        # os.remove(file_name)
    gif_path = os.path.join(video_dir, 'results_{}_{}_{}.gif'.format(opt.data_mode, epoch, seed))
    imageio.mimsave(gif_path, images, fps=20)
    # optimize(gif_path)

    '''
    os.chdir(CONF.PATH.OUTPUT)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    '''

import torch.nn as nn
def plot_classifier(model, max, min):
    max = max.values.detach().numpy()
    min = min.values.detach().numpy()

    w = 0
    for layer in model.children():
        if isinstance(layer, nn.Linear):
            w = layer.state_dict()['weight'].cpu().numpy()

    slope = (-w[0, 0]/w[0, 1] - 1) / (1 + w[0, 1]/w[0, 0])

    x = np.linspace(min-0.5, max+0.5, 100)
    y = slope * x
    return x, y
