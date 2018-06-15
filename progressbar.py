# -*- coding: utf-8 -*-
import sys 
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s'  % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    # print '\r{0} |{1}| {2}%% {3}'.format(prefix, bar, percent, suffix),
    # print('\r%s |%s| %s%% %s\r' % (prefix, bar, percent, suffix), )
    # Print New Line on Complete
    if iteration == total: 
        print(" ")