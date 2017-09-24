import IMTreatment.file_operation as imtio
import numpy as np


def parametric_test(func, kwargs, update=False):
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
                "y", "z"]
    fun_name = func.__name__
    for let, kwarg in zip(alphabet, kwargs):
        filename = f"test_{fun_name}_{let}.cimt"
        res = func(**kwarg)
        if update:
            imtio.export_to_file(res, filename)
        res2 = imtio.import_from_file(filename)
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
