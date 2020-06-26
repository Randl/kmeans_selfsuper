import matplotlib.pyplot as plt


def confusion_mat(C, row_ind, col_ind, name=''):
    conf_mat = C[row_ind][:, col_ind]

    fig = plt.figure(figsize=(9, 7))
    im = plt.imshow(conf_mat)

    plt.title("Confusion matrix")
    fig.tight_layout()
    plt.savefig(name + '_conf.pdf')
