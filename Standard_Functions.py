import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_history(history):
    """
    Code adapted from: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    by Jason Brownlee

    Function that displays the accuracy and validation loss graphs for a trained model.
    """
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 50

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(x, acc, 'b', label='Training accuracy')
    plt.plot(x, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy', fontsize=22)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss', fontsize=22)
    plt.legend()


''' Many of these functions are extraneous and have already been created by libraries such as
    sklearn. I highly suggest that you use the perfomance metrics functions created by sklearn.
    The reason I hardcoded many of these functions was so that I would have more flexibility in
    outputting my results. However, in most cases it is much easier to simply use output processing functions
    created by other people. '''


def accuracy_per_class(y_pred, y_test, columns_list):
    """ Assumes y_pred and y_test are identical shape and are both one hot encoded. Both y_pred
    and y_test should be numpy arrays. Columns_list should just be a list of columns.
    Dict format: {class: [correct, incorrect]} """
    accuracy_dict = {}
    correct_index = 0
    incorrect_index = 1
    for row in range(y_pred.shape[0]):
        curr_correct_index = np.argmax(y_test[row])
        curr_col_name = columns_list[curr_correct_index]
        if np.argmax(y_pred[row]) == curr_correct_index:
            if curr_col_name in accuracy_dict:
                accuracy_dict[curr_col_name][correct_index] += 1
            else:
                accuracy_dict[curr_col_name] = [1, 0]
        elif np.argmax(y_pred[row]) != curr_correct_index:
            if curr_col_name in accuracy_dict:
                accuracy_dict[curr_col_name][incorrect_index] += 1
            else:
                accuracy_dict[curr_col_name] = [0, 1]
    for class_name in accuracy_dict:
        lst = accuracy_dict[class_name]
        percentage = lst[correct_index] / (lst[incorrect_index]+lst[correct_index])
        accuracy_dict[class_name].insert(0, percentage)
    return accuracy_dict


def accuracy_per_class_binary(y_pred, y_test):
    """ [correct, incorrect] """
    accuracy_dict = {'normal': [0,0], 'anomaly': [0,0]}
    for row in range(y_pred.shape[0]):
        target_output = y_test[row]
        predicted_output = np.around(y_pred[row], decimals=0)
        if target_output == 0:
            if predicted_output != target_output:
                accuracy_dict['normal'][1] += 1
            elif predicted_output == target_output:
                accuracy_dict['normal'][0] += 1
        if target_output == 1:
            if predicted_output != target_output:
                accuracy_dict['anomaly'][1] += 1
            elif predicted_output == target_output:
                accuracy_dict['anomaly'][0] += 1
    for class_name in accuracy_dict:
        lst = accuracy_dict[class_name]
        percentage = lst[0]/(lst[1]+lst[0])
        accuracy_dict[class_name].insert(0, percentage)
    return accuracy_dict


def tp_tn_fp_fn(accuracy_foreach_class_dict):
    """ The dictionary I defined previously. """
    tp = accuracy_foreach_class_dict["anomaly"][1]
    tn = accuracy_foreach_class_dict["normal"][1]
    fp = accuracy_foreach_class_dict["normal"][2]
    fn = accuracy_foreach_class_dict["anomaly"][2]
    return tp, tn, fp, fn


''' Functions to calculate and display a confusion matrix. You can find other ways to 
    present your results if needed. '''


def confusion_matrix_type(y_pred, y_test):
    confusion_matrix_dict = {
        'normal': [0, 0, 0, 0, 0],
        'dos': [0, 0, 0, 0, 0],
        'r2l': [0, 0, 0, 0, 0],
        'u2r': [0, 0, 0, 0, 0],
        'probe': [0, 0, 0, 0, 0]
    }
    index_to_type_dict = {
        0: 'normal',
        1: 'dos',
        2: 'r2l',
        3: 'u2r',
        4: 'probe'
    }
    for row in range(y_pred.shape[0]):
        target_output = y_test[row]
        target_index = np.argmax(target_output)
        predicted_output = y_pred[row]
        predicted_index = np.argmax(predicted_output)

        target_type = index_to_type_dict[target_index]
        predicted_type = index_to_type_dict[predicted_index]

        confusion_matrix_dict[target_type][predicted_index] += 1

    return confusion_matrix_dict


def show_confusion_matrix(confusion_matrix_dict):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 50

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    index_to_type_dict = {
        0: 'normal',
        1: 'dos',
        2: 'r2l',
        3: 'u2r',
        4: 'probe'
    }
    ax = plt.subplot()
    confusion_matrix_array = []
    for i in range(5):
        the_type = index_to_type_dict[i]
        confusion_matrix_array.append(confusion_matrix_dict[the_type])
    confusion_matrix_array = np.asarray(confusion_matrix_array)
    confusion_matrix_array = normalize(confusion_matrix_array, axis=1, norm='l2')
    sns.heatmap(confusion_matrix_array, annot=True, ax=ax, cmap='Reds')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix', fontsize=23)
    ax.xaxis.set_ticklabels(['normal', 'dos', 'r2l', 'u2r', 'probe'])
    ax.yaxis.set_ticklabels(['probe', 'u2r', 'r2l', 'dos', 'normal'])
    # plt.matshow(confusion_matrix_array)
    # plt.colorbar()
    plt.show()