import warnings


def compare_fs(ground_fs, target_fs):
    ground_fs_set = ground_fs
    target_fs_set = target_fs
    tp = ground_fs_set & target_fs_set
    fp = target_fs_set - ground_fs_set
    fn = ground_fs_set - target_fs_set
    return cal_score(len(tp), len(fp), len(fn))


def _compare_fb(entry, fb, other_fbs):
    if entry not in other_fbs:
        return False
    elif fb != other_fbs[entry]:
        return False
    return True


def compare_fb(ground_fb, target_fb):
    tp = fp = fn = 0
    for entry, fb in ground_fb.items():
        if not _compare_fb(entry, fb, target_fb):
            fn += 1
        else:
            tp += 1

    for entry, fb in target_fb.items():
        if not _compare_fb(entry, fb, ground_fb):
            fp += 1

    return cal_score(tp, fp, fn)


def cal_score(tp, fp, fn):
    p, warn_p = precision_score(tp, fp)

    if warn_p:
        warnings.warn("Precision: Div by Zero")

    r, warn_r = recall_score(tp, fn)
    if warn_r:
        warnings.warn("Recall: Div by Zero")

    f1, warn_f1 = f1_score(p, r)
    if warn_f1:
        warnings.warn("F1-Score: Div by Zero")

    return f1, p, r


def precision_score(tp, fp):
    if fp == 0:
        return 1, ""
    # if tp + fp == 0:
    #     return 0, "warn"
    else:
        return tp / (tp + fp), ""


def recall_score(tp, fn):
    if fn == 0:
        return 1, ""
    # if tp + fn == 0:
    #     return 0, "warn"
    else:
        return tp / (fn + tp), ""


def f1_score(p, r):
    if p + r == 0:
        return 0, "warn"
    else:
        return 2 * p * r / (p + r), ""
