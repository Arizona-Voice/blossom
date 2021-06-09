# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"Error in import matplotlib{e}")

def print_free_style(message, print_fun=print):
    print_fun("▓  {}".format(message))
    print_fun("")

def print_time_style(message, print_fun=print):
    print_fun("")
    print_fun("⏰  {}".format(message))
    print_fun("")
    
def print_warning_style(message, print_fun=print):
    print_fun("")
    print_fun("⛔️  {}".format(message))
    print_fun("")
    
def print_notice_style(message, print_fun=print):
    print_fun("")
    print_fun("📌  {}".format(message))
    print_fun("")

def print_line(text, print_fun=print):
    print_fun("")
    print_fun("➖➖➖ {} ➖➖➖".format(text.upper()))
    print_fun("")


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, 
                          save_dir=None):


    """Function to plot confusion matrics.

    :param cm: confusion_matrix: function in sklearn.
    :param target_names: list of classes.
    :param cmap: str or matplotlib Colormap: Colormap recognized by matplotlib.
    :param normalize: normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population.
    :param save_dir: str: directory address to save.
    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    width = int(10)
    height = int(8)

    plt.figure(figsize=(width, height))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label. Metrics: accuracy={:0.4f}; misclass={:0.4f} \n\n'.format(accuracy, misclass))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        print(f"Save confusion-matrix...")
        plt.savefig((save_dir + '/{}.png'.format(title)))
    except IOError:
        print(f"Could not save file in directory: {save_dir}")