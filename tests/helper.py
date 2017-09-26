import IMTreatment.file_operation as imtio
import numpy as np


def sane_parameters():
    pass


def parametric_test(func, kwargs, update=False):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
                "y", "z"]
    fun_name = func.__name__
    for let, kwarg in zip(alphabet, kwargs):
        filename = "test_{}_{}.cimt".format(fun_name, let)
        res = func(**kwarg)
        # Update if necessary
        if update:
            imtio.export_to_file(res, filename)
        # If the file is not present, create it
        try:
            res2 = imtio.import_from_file(filename)
        except FileNotFoundError:
            print("file '{}' is not present (normal for a first run),"
                  "I created it for you !".format(filename))
            imtio.export_to_file(res, filename)
            continue
        # Else try recursively to test for equality
        try:
            res[0][0]
            for r, r2 in zip(res, res2):
                assert np.all(r == r2)
        except TypeError:
            try:
                res[0]
                assert np.all(res == res2)
            except TypeError:
                assert res == res2
