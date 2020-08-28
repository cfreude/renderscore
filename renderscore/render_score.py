import numpy as np
from scipy.stats import norm


def score_func(a_m, b_m, a_s, b_s):
    m_d = a_m - b_m

    # both have zero STD
    if a_s <= 0.0 and b_s <= 0.0:
        if m_d == 0.0:
            return np.inf
        else:
            return 0
    elif a_s <= 0.0 or b_s <= 0.0:
        return np.nan

    a_s2 = a_s * a_s
    b_s2 = b_s * b_s

    c1 = -(m_d * m_d / (2.0 * (a_s2 + b_s2)))
    c2 = np.sqrt(2.0 * np.pi, dtype=np.float64) * a_s * b_s * np.sqrt(1.0 / a_s2 + 1.0 / b_s2, dtype=np.float64)

    return np.exp(c1) / c2


def score_func_array(a_m, b_m, a_s, b_s):
    m_d = a_m - b_m

    both_std_zero = np.logical_and(a_s <= 0.0, b_s <= 0.0)

    inf_mask = np.logical_and(both_std_zero, m_d <= 0.0)
    zero_mask = np.logical_and(both_std_zero, m_d > 0.0)
    nan_mask = np.logical_and(np.logical_or(a_s <= 0.0, b_s <= 0.0), np.logical_not(both_std_zero))

    fixed_a_s = np.where(a_s > 0, a_s, 1.0)
    fixed_b_s = np.where(b_s > 0, b_s, 1.0)

    '''
    # both have zero STD
    if a_s <= 0.0 and b_s <= 0.0:
        if m_d == 0.0:
            return np.inf
        else:
            return 0
    elif a_s <= 0.0 or b_s <= 0.0:
        return np.nan
    '''

    a_s2 = np.multiply(fixed_a_s, fixed_a_s)
    b_s2 = np.multiply(fixed_b_s, fixed_b_s)

    c1 = np.divide(-np.multiply(m_d, m_d), 2.0 * (a_s2 + b_s2))
    c2 = np.sqrt(2.0 * np.pi, dtype=np.float64) * np.multiply(fixed_a_s, fixed_b_s) * \
        np.sqrt(1.0 / a_s2 + 1.0 / b_s2, dtype=np.float64)

    score = np.exp(c1) / c2

    score[inf_mask] = np.inf
    score[zero_mask] = 0.0
    score[nan_mask] = np.nan

    return score


def normality_factor(samples, ks=False):
    N = len(samples)
    M = np.mean(samples, dtype=np.float64)
    S = np.std(samples, ddof=1, dtype=np.float64)

    if S <= 0:
        return 1.0, ([M, M], [1.0, 1.0], [1.0, 1.0])

    ecdf_x = np.array(np.sort(samples), dtype=np.float64)
    ecdf_y_p = np.arange(1.0, N + 1, dtype=np.float64) / N
    ecdf_y_m = np.arange(0.0, N, dtype=np.float64) / N
    cdf_y = norm(M, S).cdf(ecdf_x)

    Dplus = (ecdf_y_p - cdf_y).max()
    Dmin = (cdf_y - ecdf_y_m).max()
    if ks:
        D = np.max([Dplus, Dmin])
    else:
        D = Dplus + Dmin
    return 1.0 - D, (ecdf_x, ecdf_y_m, cdf_y)


def normality_factor_array(samples, ks=False):
    N = samples.shape[0]
    M = np.mean(samples, axis=0, dtype=np.float64)
    S = np.std(samples, axis=0, ddof=1, dtype=np.float64)

    valid_std_mask = S > 0
    valid_std_mask_rep = np.repeat(np.expand_dims(valid_std_mask, 0), N, axis=0)
    fixed_samples = np.where(valid_std_mask_rep, samples, 1.0)
    fixed_S = np.where(valid_std_mask_rep, S, 1.0)

    # if S <= 0:
    #    return 1.0, ([M, M], [1.0, 1.0], [1.0, 1.0])

    ecdf_x = np.sort(fixed_samples, axis=0).astype(np.float64)
    ecdf_y_p = np.arange(1.0, N + 1, dtype=np.float64) / N
    for i in range(1, len(samples.shape)):
        ecdf_y_p = np.expand_dims(ecdf_y_p, i)
        ecdf_y_p = np.repeat(ecdf_y_p, samples.shape[i], axis=i)

    ecdf_y_m = np.arange(0.0, N, dtype=np.float64) / N
    for i in range(1, len(samples.shape)):
        ecdf_y_m = np.expand_dims(ecdf_y_m, i)
        ecdf_y_m = np.repeat(ecdf_y_m, samples.shape[i], axis=i)
    cdf_y = norm(M, fixed_S).cdf(ecdf_x)

    Dplus = (ecdf_y_p - cdf_y).max(axis=0)
    Dmin = (cdf_y - ecdf_y_m).max(axis=0)

    if ks:
        D = np.max([Dplus, Dmin], axis=0)
    else:
        D = Dplus + Dmin

    D[~valid_std_mask] = 1.0

    return 1.0 - D


def compute_score(a_samp, b_samp):
    img_shape = a_samp.shape[1:]

    score_img = np.zeros(img_shape, dtype=np.float64)
    a_ks = np.zeros(img_shape, dtype=np.float64)
    b_ks = np.zeros(img_shape, dtype=np.float64)
    a_m_img = np.zeros(img_shape, dtype=np.float64)
    b_m_img = np.zeros(img_shape, dtype=np.float64)
    a_s_img = np.zeros(img_shape, dtype=np.float64)
    b_s_img = np.zeros(img_shape, dtype=np.float64)

    for pixel in np.ndindex(img_shape):
        data_index = (slice(None),) + pixel

        a_samp_temp = a_samp[data_index]
        b_samp_temp = b_samp[data_index]

        a_m = np.mean(a_samp_temp, dtype=np.float64)
        a_s = np.std(a_samp_temp, ddof=1, dtype=np.float64)

        b_m = np.mean(b_samp_temp, dtype=np.float64)
        b_s = np.std(b_samp_temp, ddof=1, dtype=np.float64)

        a_m_img[pixel] = a_m
        b_m_img[pixel] = b_m
        a_s_img[pixel] = a_s
        b_s_img[pixel] = b_s

        a_ks[pixel] = normality_factor(a_samp_temp)[0]  # if a_s > 0.0 else 1.0
        b_ks[pixel] = normality_factor(b_samp_temp)[0]  # if b_s > 0.0 else 1.0

        score_img[pixel] = score_func(a_m, b_m, a_s, b_s)  # if a_s > 0.0 and b_s > 0.0 else 0.0

    return score_img


def test_score_array():
    import time
    s = time.time()
    np.random.seed(0)
    AM = np.random.rand(100, 32, 32, 3)
    BM = np.random.rand(100, 32, 32, 3)
    AS = np.random.rand(100, 32, 32, 3)
    BS = np.random.rand(100, 32, 32, 3)
    # result_par = score_func_array(AM, BM, AS, BS); quit()

    result = np.zeros(AM.shape)
    for ind in np.ndindex(AM.shape):
        result[ind] = score_func(AM[ind], BM[ind], AS[ind], BS[ind])
    t1 = time.time() - s
    print 'time (loop): %.4f' % t1
    # print result

    s = time.time()
    result_par = score_func_array(AM, BM, AS, BS)
    t2 = time.time() - s
    print 'time (matrix): %.4f' % t2
    print 'time fac: %.2f' % (t1 / t2)
    delta = result - result_par
    delta = delta[np.isfinite(delta)]
    print 'error sum:', np.sum(delta), len(delta)


def test_norm_array():
    import time
    s = time.time()
    np.random.seed(0)
    data = np.random.rand(100, 32, 32, 3)
    print 'data shape:', data.shape
    result = np.zeros(data.shape[1:])
    for ind in np.ndindex(data.shape[1:]):
        samples = data[(slice(None),) + ind]
        result[ind] = normality_factor(samples)[0]
    t1 = time.time() - s
    print 'time (loop): %.4f' % t1
    # print result

    s = time.time()
    result_par = normality_factor_array(data)
    t2 = time.time() - s
    print 'time (matrix): %.4f' % t2
    print 'time fac: %.2f' % (t1 / t2)
    print 'error sum:', np.sum(result_par - result)


if __name__ == "__main__":
    test_norm_array()
    test_score_array()
