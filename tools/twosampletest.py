#!/usr/bin/env python
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
############################################################
#
# pylint: disable=too-many-locals,invalid-name,line-too-long,consider-using-f-string
# pylint: disable=too-many-statements
'''
Peform two-sample mean t- or z-test
'''

import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats._result_classes import TtestResult
from scipy.stats import distributions
from scipy import stats


def _equal_var_ttest_denom(v1, n1, v2, n2):
    """t-test for equal variance"""
    # If there is a single observation in one sample, this formula for pooled
    # variance breaks down because the variance of that sample is undefined.
    # The pooled variance is still defined, though, because the (n-1) in the
    # numerator should cancel with the (n-1) in the denominator, leaving only
    # the sum of squared differences from the mean: zero.
    v1 = np.where(n1 == 1, 0, v1)[()]
    v2 = np.where(n2 == 1, 0, v2)[()]

    df = n1 + n2 - 2.0
    svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    return df, denom


def _get_pvalue(statistic, distribution, alternative, symmetric=True):
    """Get p-value given the statistic, (continuous) distribution, and alternative"""

    if alternative == 'less':
        pvalue = distribution.cdf(statistic)
    elif alternative == 'greater':
        pvalue = distribution.sf(statistic)
    elif alternative == 'two-sided':
        pvalue = 2 * (distribution.sf(np.abs(statistic)) if symmetric
                      else np.minimum(distribution.cdf(statistic),
                                      distribution.sf(statistic)))
    else:
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue


def _ttest_ind_from_stats(mean1, mean2, denom, df, alternative):
    """t-test ..."""
    d = mean1 - mean2
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, denom)[()]
    prob = _get_pvalue(t, distributions.t(df), alternative)

    return (t, prob)


def t_test(data, alpha=0.95, output=True):
    '''
    Perform t-test on the means of two groups -- on per line of data
    '''

    results = []
    table = None
    for row in data.index:
        n1 = data["na"][row]
        n2 = data["nb"][row]
        m1 = data["meana"][row]
        m2 = data["meanb"][row]
        s1 = data["sda"][row]
        s2 = data["sdb"][row]
        err = m1 - m2
        df = n1 + n2 - 2
        Sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/df)
        SE = Sp * np.sqrt(1/n1 + 1/n2)
        alternative = "two-sided"

        # do a t-Student test
        # ttest = stats.ttest_ind(aok, bok, equal_var=True, alternative='two-sided')
        v1 = s1**2
        v2 = s2**2
        df, denom = _equal_var_ttest_denom(v1, n1, v2, n2)
        t, prob = _ttest_ind_from_stats(m1, m2, denom, df, alternative)
        # when nan_policy='omit', `df` can be different for different axis-slices
        df = np.broadcast_to(df, t.shape)[()]
        estimate = m1-m2
        # _axis_nan_policy decorator doesn't play well with strings
        alternative_num = {"less": -1,
                           "two-sided": 0, "greater": 1}[alternative]
        ttest = TtestResult(t, prob, df=df, alternative=alternative_num,
                            standard_error=denom, estimate=estimate)
        t = stats.t.interval(confidence=alpha, df=df)
        ci = err + np.array(t) * SE
        clabel = 't-Student'
        rlabel = 't-stats(df={})'.format(df)

        # Large P-value we do not reject H0: A=B
        if ttest.pvalue > 1-alpha:
            result = "same"
        else:
            result = "DIFF"
        if output:
            stab = pd.DataFrame({'Stats': ['mean', 'stdev', 'Err(A-B)', rlabel, 'P-value', 'CI({}) L'.format(alpha), 'CI({}) U'.format(alpha)],
                                 'Grp A(n={})'.format(n1): [m1, s1, None, None, None, None, None],
                                 'Grp B(n={})'.format(n2): [m2, s2, None, None, None, None, None],
                                 clabel: [None, None, err, ttest.statistic, ttest.pvalue, ci[0], ci[1]]})
            print("* {},{}".format(data["pname"][row], result))
            print("H0: mean(A) = mean(B)\nH1: mean(A) /= mean(B)\n")
            # print("P-value greater than alpha indicates not enough evidence to reject H0,\nand take that the means are indistinguishable\n")
            print(stab.to_markdown(index="never").replace("nan", "   "))

        mok = True
        if m1 == 0.0:
            print('warning: mean for group A is zero, solve bigger problem(s)')
            mok = False

        if m2 == 0.0:
            print('warning: for group B is zero, solve bigger problem(s)')
            mok = False

        if mok:
            speedup = m1/m2
        else:
            speedup = float('nan')
            print('warning: setting speedup to NaN')

        results.append(result)
        if table is None:
            table = pd.DataFrame({'result': result, 'p-val': ttest.pvalue, 'low': ci[0], 'upp': ci[1],
                                  'sgnc': 1-alpha, 'dif': estimate, 'speedup': speedup}, index=[row])
        else:
            table = pd.concat([table, pd.DataFrame({'result': result, 'p-val': ttest.pvalue, 'low': ci[0], 'upp': ci[1],
                               'sgnc': 1-alpha, 'dif': estimate, 'speedup': speedup}, index=[row])])

    return results, table


def main(argv):
    '''Main entry point for two-sample t-test computatioss'''

    parser = argparse.ArgumentParser(
        description="Perform a two-sample t-test")
    _ = parser.add_argument('-f', '-i', '--filename', '--input', '-csv',
                            type=argparse.FileType('r'), required=True, help="csv filename to process")
    _ = parser.add_argument(
        '-c', '--confidence', help="define test significance (alpha), min: 0.001, max: 0.999", default=0.95, type=float)

    args = parser.parse_args(argv)
    file = args.filename.name

    alpha = args.confidence
    if alpha < 0.001 or alpha > 0.999:
        print("Warning: the provided alpha value is too extreme, setting to 0.95\n")
        alpha = 0.95

    try:
        d = pd.read_csv(file, header=None, names=[
                        "pname", "meana", "sda", "na", "meanb", "sdb", "nb"])
    except FileNotFoundError:
        print("Error: something bad occurred while opening or reading the input file: " + file + "...")
        sys.exit(4)

    t_test(data=d, alpha=alpha)


if __name__ == "__main__":
    main(sys.argv[1:])
