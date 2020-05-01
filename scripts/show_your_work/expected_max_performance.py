import scipy
import numpy as np

def _cdf_with_replacement(i,n,N):
    return (i/N)**n

def _compute_variance(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of 
    http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    """
    variance_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_var = 0
        for i in range(N):
            cur_var += (cur_data[i] - expected_max_cond_n[n])**2 * pdfs[n][i]
        cur_var = np.sqrt(cur_var)
        variance_of_max_cond_n.append(cur_var)
    return variance_of_max_cond_n
    

# this implementation assumes sampling with replacement for computing the empirical cdf
def samplemax(validation_performance):
    validation_performance = list(validation_performance)
    validation_performance.sort()
    N = len(validation_performance)
    pdfs = []
    for n in range(1,N+1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1,N+1):
            F_Y_of_y.append(_cdf_with_replacement(i,n,N))


        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]
        
        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += validation_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)


    var_of_max_cond_n = _compute_variance(N, validation_performance, expected_max_cond_n, pdfs)

    return {"mean":expected_max_cond_n, "var":var_of_max_cond_n, "max": np.max(validation_performance),
            "min":np.min(validation_performance)}

if __name__ == "__main__":
    example_valid_perf = np.random.uniform(0,1, 20)
    print(samplemax(example_valid_perf))
