import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_function(loss, accuracy, result_dir):

    plt.plot(loss['mapping'])
    plt.plot(loss['adversarial'])
    plt.plot(loss['class'])
    plt.plot(loss['val_mapping'])
    plt.plot(loss['val_adversarial'])
    plt.plot(loss['val_class'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['mapping', 'adversarial', 'class', 'val_mapping', 'val_adversarial', 'val_class'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'loss_plot.png'), dpi=600)
    plt.close()

    plt.plot(accuracy['source'])
    plt.plot(accuracy['target'])
    plt.plot(accuracy['weighted'])
    plt.plot(accuracy['val_source'])
    plt.plot(accuracy['val_target'])
    plt.plot(accuracy['val_weighted'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['source', 'target', 'weighted', 'val_source', 'val_target', 'val_weighted'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'accuracy_plot.png'), dpi=600)
    plt.close()
