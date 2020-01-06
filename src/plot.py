import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import squarify
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine

import preprocess


def plot_line(title, x_label, y_label, x, y_values):
    fig, ax = plt.subplots()

    ax.plot(x, y_values)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid()
    ax.set_title(title)

    plt.draw()


def plot_lines(title, legend_title, y_label, x, labels, y_values, x_label):
    if len(y_values) != len(labels):
        print("\033[1;31m The length of labels and arguments must be the same. Arguments has length", len(y_values),
              "and Labels has length", len(labels), "\033[0;0m")
        return
    fig, ax = plt.subplots()
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

    for i, (arg, label) in enumerate(zip(y_values, labels)):
        ax.plot(x, arg, label=label, marker=filled_markers[i % len(filled_markers)])

    ax.legend(title=legend_title)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, np.amax(y_values) + 0.1)
    ax.set_xlabel(x_label)
    ax.grid()
    ax.set_title(title)
    plt.draw()


def plot_two_subplotted_lines(title, x1_label, y1_label, x1, y1_values, x2_label, y2_label, x2, y2_values):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x1, y1_values)

    axs[0].set_xlabel(x1_label)
    axs[0].set_ylabel(y1_label)
    axs[0].grid()

    axs[1].plot(x2, y2_values)

    axs[1].set_xlabel(x2_label)
    axs[1].set_ylabel(y2_label)
    axs[1].grid()

    fig.suptitle(title)

    plt.draw()


def plot_confusion_matrix(
        cm,
        classes,
        predicted_classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, predicted_classes, rotation="vertical")
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()


def plot_half_matrix(cm,
                     classes,
                     predicted_classes,
                     title='Confusion matrix',
                     cmap=plt.cm.Blues):
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    fig, ax = plt.subplots(figsize=(8, 8))
    cm = np.asarray(cm)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i < j:
            cm[j, i] = 0
    im = ax.imshow(cm, cmap=cmap)
    ax.set_title(title + "\n")

    ax.set_xticklabels(classes)
    ax.set_yticklabels(predicted_classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i <= j:
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi / 2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts


def radar_chart(data, spoke_labels, N, labels):
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                           subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['b', 'r', 'g', 'm']
    # Plot the four cases from the example data on separate axes
    (title, case_data) = data
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plot
    legend = ax.legend(labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios',
             horizontalalignment='center', color='black', weight='bold',
             size='large')


def plot_3d_scatter(dataset, title):
    aux_df = dataset.iloc[:, :-1].copy()
    aux_df = preprocess.preprocess_for_plot(aux_df)
    aux_df["Classes"] = [value.decode("utf-8") for value in dataset.iloc[:, -1]]
    columns = np.asarray(aux_df.columns.values)

    sns.set_palette(sns.color_palette())
    pp = sns.pairplot(aux_df, vars=columns[:-1], hue=columns[-1], height=1.8, aspect=2,
                      plot_kws=dict(s=25, edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
    pp._legend.remove()
    pp.fig.set_figheight(6)
    pp.fig.set_figwidth(12)
    pp.fig.legend(handles=pp._legend_data.values(),
                  labels=pp._legend_data.keys(),
                  title="Classes",
                  loc="center right")

    pp.fig.subplots_adjust(top=0.93, wspace=0.3)
    pp.fig.suptitle(title)
    pp.fig.canvas.set_window_title(title)


def plot_prop_variance(all_components, povs, n_components):
    plt.figure(figsize=(10, 5))
    plt.plot(all_components, povs)
    plt.yticks(list(plt.yticks()[0]) + [povs[n_components - 1]])
    plt.xticks(list(plt.xticks()[0]) + [n_components])
    plt.axhline(povs[n_components - 1], color='r', linestyle='--', alpha=0.4)
    plt.axvline(n_components, color='r', linestyle='--', alpha=0.4)
    plt.plot(n_components, povs[n_components - 1], 'ro', markersize=5)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative variance explained')
    plt.title('Cumulative variance explained')


def heatmap(data, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    plt.figure(figsize=(12, 6))

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, aspect="auto")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_heatmap(transform_matrix_reduced, x_label, y_label, title):
    final_matrix = np.copy(transform_matrix_reduced)
    if final_matrix.shape[0] > 15:
        final_matrix = final_matrix[:15, :]
        print("\033[93mWarning in \"plot.plot_matrix\": Reshaping X axis features (max. 15) to plot matrix.\033[0m")
    if final_matrix.shape[1] > 15:
        final_matrix = final_matrix[:, :15]
        print("\033[93mWarning in \"plot.plot_matrix\": Reshaping Y axis features (max. 15) to plot matrix.\033[0m")
    im, cbar = heatmap(final_matrix, cmap="Blues")
    texts = annotate_heatmap(im, valfmt="{x:.3f}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def plot_sofm(data_array, sofmnet):
    plt.figure(figsize=(12, 6))
    plt.plot(data_array.T[0:1, :], data_array.T[1:2, :], 'ko')
    plt.xlim(-1, 1.2)
    plt.ylim(-1, 1.2)

    plt.plot(sofmnet.weight, 'bx', markersize=20)


def plot_scatter(x, y, labels, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(x)):
        ax.scatter(x[i], y[i], label=labels[i])

    plt.xlim(left=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend()


def plot_treemap(reduction_percentages, labels, title):
    colors = [plt.cm.Spectral(i / float(len(reduction_percentages))) for i in range(len(reduction_percentages))]
    plt.figure(figsize=(9, 8), dpi=120)
    squarify.plot(sizes=reduction_percentages, label=labels, alpha=.6, color=colors)
    plt.title(title, fontsize=16, fontweight="medium")
    plt.axis('off')


def plot_bubblechart(accuracies, execution_times, labels, xlabel, ylabel, title):
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    colors = [plt.cm.Spectral(i / float(len(accuracies))) for i in range(len(accuracies))]
    x = np.arange(len(accuracies))
    labels_mod = list(map(lambda x: str(x[0]) + "\n (" + str(round(x[1], 2)) + "s)", zip(labels, execution_times)))
    execution_times = np.divide(execution_times, max(execution_times))
    plt.figure(figsize=(9, 8), dpi=120)
    plt.title(title, fontsize=14, fontweight="medium")
    plt.scatter(x, accuracies, s=execution_times * 30000, c=colors, alpha=0.6)
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel(ylabel)
    for line, acc in zip(range(0, len(labels)), accuracies):
        plt.text(x[line], acc - 0.03, labels_mod[line], horizontalalignment='center', size='medium', color='black',
                 weight='semibold')


# def plot_bubblechart(accuracies, execution_times, reductions, labels, xlabel, ylabel, title ):
#     plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
#     plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
#     colors = [plt.cm.Spectral(i / float(len(accuracies))) for i in range(len(accuracies))]
#     x = np.arange(len(accuracies))
#     labels_mod = list(map(lambda x: str(x[0]) + "\n (" + str(round(x[1], 2)) + "s)", zip(labels, execution_times)))
#     execution_times_normalized = np.divide(execution_times, max(execution_times))
#     plt.figure(figsize=(9, 8), dpi=120)
#     plt.title(title, fontsize=16, fontweight="medium")
#
#     plt.scatter(reductions, accuracies, s=execution_times_normalized*30000, c=colors, alpha=0.6)
#     plt.xlabel(xlabel)
#     #plt.xticks(np.arange(len(labels)), labels)
#     plt.ylabel(ylabel)
#     for line, acc in zip(range(0, len(labels)), accuracies):
#         plt.text(reductions[line], acc - 0.03, labels_mod[line], horizontalalignment='center', size='medium', color='black',
#                 weight='semibold')


def plot_bar(x, ys, col_labels, x_ticks, title):
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    x_bar = x - width / 2
    for y, col_labels in zip(ys, col_labels):
        ax_rects = ax.bar(x_bar, y, width, label=col_labels)
        autolabel(ax_rects)
        x_bar += width
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_ticks)
    ax.legend()

    fig.tight_layout()


def plot_histograms(values, titles):
    fig, axs = plt.subplots(2, 3, tight_layout=True)
    if len(values) == len(titles):
        for i, ax in enumerate(axs.flat):
            ax.hist(values[i], bins=30)
            ax.set_title(titles[i])


def show_all():
    plt.show()
