import os

def print_bar (iteration: int, total: int, prefix = '', suffix = '', decimals = 1, length = "fit", fill = '█') -> None:
    """Prints a progress bar on the terminal

    Example
    -------
    for i in range(50):
        print_bar(i+1, 50, prefix="simple loop:", length=40)
    """
    if length=="fit":
        rows, columns = os.popen('stty size', 'r').read().split() # checks how wide the terminal width is
        length = int(columns) // 2
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: # go to new line when the progress bar is finished
        print()
