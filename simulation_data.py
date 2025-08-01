import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from pathlib import Path
from GENetLib.fda_func import bspline_mat
from GENetLib.fda_func import create_bspline_basis
from GENetLib.fda_func import fd
from GENetLib.fda_func import eval_fd

from AuxFunc import AuxFunc


def pf_sim_data(n, T, nknots, order, seed = 123):
    np.random.seed(seed)
    q = 9
    class1 = np.arange(n/3)
    class2 = np.arange(n/3, n/3*2)
    class3 = np.arange(n/3*2, n)
    t = np.linspace(0, 1, T)
    c1_1 = np.cos(2 * np.pi * t)
    c1_2 = 1 + np.sin(2 * np.pi * t)
    c1_3 = 2 * (np.sin(2 * np.pi * t) + np.cos(2 * np.pi * t))
    c2_1 = 1 - 2 * np.exp(-6 * t)
    c2_2 = 2 * t**2
    c2_3 = 1 + t**3
    c3_1 = -1.5 * t
    c3_2 = t + 1
    c3_3 = 2 * np.sqrt(t) + 1
    sigma = 0.6
    oridata_list = [[] for _ in range(n)]
    for i in range(n):
        if i in class1:
            oridata_list[i] = [
                c1_1 + np.random.normal(0, sigma, T),
                c1_1 + np.random.normal(0, sigma, T),
                c1_1 + np.random.normal(0, sigma, T),
                c1_2 + np.random.normal(0, sigma, T),
                c1_2 + np.random.normal(0, sigma, T),
                c1_2 + np.random.normal(0, sigma, T),
                c1_3 + np.random.normal(0, sigma, T),
                c1_3 + np.random.normal(0, sigma, T),
                c1_3 + np.random.normal(0, sigma, T)
            ]
        elif i in class2:
            oridata_list[i] = [
                c2_1 + np.random.normal(0, sigma, T),
                c2_1 + np.random.normal(0, sigma, T),
                c2_1 + np.random.normal(0, sigma, T),
                c2_2 + np.random.normal(0, sigma, T),
                c2_2 + np.random.normal(0, sigma, T),
                c2_2 + np.random.normal(0, sigma, T),
                c2_3 + np.random.normal(0, sigma, T),
                c2_3 + np.random.normal(0, sigma, T),
                c2_3 + np.random.normal(0, sigma, T)
            ]
        elif i in class3:
            oridata_list[i] = [
                c3_1 + np.random.normal(0, sigma, T),
                c3_1 + np.random.normal(0, sigma, T),
                c3_1 + np.random.normal(0, sigma, T),
                c3_2 + np.random.normal(0, sigma, T),
                c3_2 + np.random.normal(0, sigma, T),
                c3_2 + np.random.normal(0, sigma, T),
                c3_3 + np.random.normal(0, sigma, T),
                c3_3 + np.random.normal(0, sigma, T),
                c3_3 + np.random.normal(0, sigma, T)
            ]
    
    # Generate a data list to contain the time information in each sample list    
    times = [[] for _ in range(n)]
    for i in range(n):
        times[i] = [t] * q
    
    # Generate a sample list to contain the all sample information in each dataframe
    sample_list = [[] for _ in range(n)]
    id_matrix = np.zeros((n, q), dtype=int)
    sample_list = []
    for i in range(n):
        for j in range(q):
            id_matrix[i, j] = len(oridata_list[i][j])
        data = {'id': np.repeat(np.arange(q), id_matrix[i, :]),
                'time': np.concatenate(times[i]),
                'y': np.concatenate(oridata_list[i])}
        df = pd.DataFrame(data)
        sample_list.append(df)
    
    # Generate a design matrix for sample basis
    timerange = np.linspace(0, 1, num = T)
    auxfunc_1 = AuxFunc(n = n, m = nknots, x = timerange)
    spline_list = []
    for i, sample in enumerate(sample_list):
        sublist = []
        for _, group in sample.groupby('id'):
            basis = bspline_mat(np.array(group['time']), auxfunc_1.knots_eq(), norder = order)
            sublist.append(basis)
        spline_list.append(sublist)
    
    # Generate response vector
    Y_list = []
    for sample in sample_list:
        Y_sublist = [np.array(sample[sample['id'] == j]['y']) for j in range(q)]
        Y_list.append(Y_sublist)
    
    # Construct missing samples under balanced data
    miss_percent = 0.3
    miss_meas = 0.2
    subgroup_sam = 3
    subgroup_fea = 3
    cluster_struc_sam = np.array([int(n/3), int(n/3), int(n/3)])
    cluster_struc_fea = np.array([3, 3, 3])
    cluster_gap_sam = np.array([0, int(n/3), int(n/3*2)])
    cluster_gap_fea = np.array([0, 3, 6])
    for i in range(subgroup_sam):
        for j in range(subgroup_fea):
            num_elem = int(cluster_struc_sam[i] * cluster_struc_fea[j])
            num_miss = int(np.ceil(num_elem * miss_percent))
            id1 = np.random.choice(num_elem, num_miss, replace=False)
            id_loac = np.zeros((num_miss, 2), dtype=int)
            mat = np.arange(num_elem).reshape(cluster_struc_sam[i], cluster_struc_fea[j])
            for l, val in enumerate(id1):
                idx = np.where(mat == val)
                id_loac[l, :] = [idx[0][0], idx[1][0]]
            for l in range(len(id1)):
                i1 = id_loac[l, 0]
                j1 = id_loac[l, 1]
                id2 = np.sort(np.random.choice(T, int((1-miss_meas)*T), replace=False))
                times[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]] = t[id2]
                basis_new = bspline_mat(t[id2], auxfunc_1.knots_eq(), norder = order)
                spline_list[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]] = basis_new
                Y_list[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]] = Y_list[i1 + cluster_gap_sam[i]][j1 + cluster_gap_fea[j]][id2]
    
    # Generate censored sample list
    censored_sample_list = []
    merged_Y_list = []
    for Y in Y_list:
        merged_Y_list.append([item for sublist in Y for item in sublist])
    for df, merged_Y in zip(sample_list, merged_Y_list):
        df_censored = df[df['y'].isin(merged_Y)]
        censored_sample_list.append(df_censored)
    
    # Generate censored sample matrix
    censored_none_list = []
    for i in range(len(censored_sample_list)):
        result_df = pd.DataFrame()
        for j in range(q):
            df = censored_sample_list[i][censored_sample_list[i]['id'] == j]
            df = df.set_index('time').reindex(timerange).reset_index()
            df['y'] = df['y'].apply(lambda x: x if not pd.isnull(x) else None)
            df['id'] = j
            result_df = pd.concat([result_df, df], ignore_index=True)
        censored_none_list.append(result_df)
    censored_sample_matrix = pd.DataFrame()
    for i in range(len(censored_none_list)):
        pivot_df = censored_none_list[i].pivot_table(index='time', columns='id', values='y')
        pivot_df.index = range(T)   
        pivot_df.reset_index(inplace=True)
        pivot_df.rename(columns={'index': 'time'}, inplace=True)
        time_index = pivot_df.columns.get_loc('time')
        pivot_df.insert(loc=time_index + 1, column = 'measurement', value = i)
        censored_sample_matrix = pd.concat([censored_sample_matrix, pivot_df], axis = 0)

    # Order and return result
    censored_sample_matrix = censored_sample_matrix.sort_values(by='time', ascending=True)
    def sort_measurement(group):
        return group.sort_values(by='measurement', ascending=True)
    censored_sample_matrix = censored_sample_matrix.groupby('time', group_keys=False).apply(sort_measurement).reset_index(drop=True)
    return {'data': censored_sample_matrix,
            'location': t,
            'feature cluster': [set(range(0,3)), set(range(3,6)), set(range(6,9))],
            'sample cluster': [set(range(0,int(n/3))), set(range(int(n/3),int(n/3*2))), set(range(int(n/3*2),n))]}
'''
pf_simdata = pf_sim_data(n = 30, T = 10, nknots = 3, order = 3, seed = 123)
'''


def local_sim_data(n, T, sigma, seed = 123):
    np.random.seed(seed)
    times = np.linspace(0, 1, T)
    class1 = np.arange(n // 2)
    class2 = np.arange(n // 2, n)
    setpoint1 = 0.6
    setpoint2 = 1
    mu = np.zeros(T)
    rho1 = 0.3
    sig = np.zeros((T, T))
    for i1 in range(T):
        for i2 in range(T):
            sig[i1, i2] = sigma ** 2 * (rho1 ** abs(i1 - i2))
    oridata_list = []
    for i in range(n):
        if i in class1:
            c1 = np.zeros(T)
            for l in range(T):
                if times[l] >= setpoint1 and times[l] <= setpoint2:
                    c1[l] = 1 + np.sin(2 * np.pi * (times[l] - setpoint1) / (setpoint2 - setpoint1))
                else:
                    c1[l] = 1
            epsilon = multivariate_normal.rvs(mean=mu, cov=sig)
            oridata_list.append(c1 + epsilon)
        elif i in class2:
            c2 = np.zeros(T)
            for l in range(T):
                if times[l] >= setpoint1 and times[l] <= setpoint2:
                    c2[l] = 1 - np.sin(2 * np.pi * (times[l] - setpoint1) / (setpoint2 - setpoint1))
                else:
                    c2[l] = 1
            epsilon = multivariate_normal.rvs(mean=mu, cov=sig)
            oridata_list.append(c2 + epsilon)
    data = pd.DataFrame(np.column_stack(oridata_list))
    return {'data': data,
            'location': times,
            'sample cluster': [set(class1), set(class2)]}
'''
local_simdata = local_sim_data(n = 100, T = 100, sigma = 0.75, seed = 42)
'''


def cc_sim_data():
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    data_path = current_dir / 'simulation_data' / 'cc_sim_data.csv'
    ccdata = pd.read_csv(data_path)
    data_dict = {group: matrix.drop(columns=["Matrix"]).values.tolist() 
                 for group, matrix in ccdata.groupby("Matrix")}
    sorted_keys = sorted([int(key.replace("Matrix", "")) - 1 for key in data_dict.keys()])
    sorted_values = [data_dict[f"Matrix{key + 1}"] for key in sorted_keys]
    cc_data = np.array(sorted_values)
    cc_data = np.transpose(cc_data, (1, 2, 0))
    return cc_data
'''
cc_simdata = cc_sim_data()
'''


def lbm_sim_data(n = 100, p = 100, t = 30, bivariate = False, noise = None, seed = 111):
    np.random.seed(seed)
    long_ = t
    x = np.linspace(0, 1, long_)
    K = 4
    L = 3
    A = np.sin(4 * np.pi * x)
    B = 0.75 - 0.5 * ((x > 0.7) & (x < 0.9)).astype(float)
    C = norm.pdf(x, loc=0.2, scale=0.02)
    C = C / C.max() if C.max() != 0 else C
    D = np.sin(10 * np.pi * x)
    fun = np.vstack([A, B, C, D])
    if bivariate:
        A2 = np.cos(4 * np.pi * x)
        B2 = 0.75 - 0.5 * ((x > 0.2) & (x < 0.4)).astype(float)
        C2 = norm.pdf(x, loc=0.2, scale=0.05)
        C2 = C2 / C2.max() if C2.max() != 0 else C2
        D2 = np.cos(10 * np.pi * x)
        fun2 = np.vstack([A2, B2, C2, D2])
    noise = 0 if bivariate else 0.1 / 3
    mu = np.full((K, L, 4), noise)
    mu[0, 0, 0] = 1 - 3 * noise;  mu[1, 0, 0] = 1 - 3 * noise;  mu[2, 0, 1] = 1 - 3 * noise;  mu[3, 0, 3] = 1 - 3 * noise
    mu[0, 1, 1] = 1 - 3 * noise;  mu[1, 1, 1] = 1 - 3 * noise;  mu[2, 1, 2] = 1 - 3 * noise;  mu[3, 1, 0] = 1 - 3 * noise
    mu[0, 2, 2] = 1 - 3 * noise;  mu[1, 2, 3] = 1 - 3 * noise;  mu[2, 2, 0] = 1 - 3 * noise;  mu[3, 2, 3] = 1 - 3 * noise
    props_rows = np.array([0.2, 0.4, 0.1, 0.3])
    counts_rows = (n * props_rows).astype(int)
    diff_n = n - counts_rows.sum()
    counts_rows[-1] += diff_n
    Z = np.concatenate([np.full(cnt, i + 1) for i, cnt in enumerate(counts_rows)])
    props_cols = np.array([0.4, 0.3, 0.3])
    counts_cols = (p * props_cols).astype(int)
    diff_p = p - counts_cols.sum()
    counts_cols[-1] += diff_p
    W = np.concatenate([np.full(cnt, i + 1) for i, cnt in enumerate(counts_cols)])
    if bivariate:
        X = np.full((n, p, long_), np.nan)
        X2 = np.full((n, p, long_), np.nan)
    else:
        X = np.full((n, p, long_), np.nan)
    Y = np.full((n, p), np.nan)
    for k in range(K):
        for l in range(L):
            rows_idx = np.where(Z == (k + 1))[0]
            cols_idx = np.where(W == (l + 1))[0]
            n_rows = len(rows_idx)
            n_cols = len(cols_idx)
            nkl = n_rows * n_cols
            if nkl == 0:
                continue
            pvals = mu[k, l, :].copy()
            if pvals.sum() > 0:
                pvals = pvals / pvals.sum()
            else:
                pvals = np.full(4, 0.25)
            draws = np.random.multinomial(1, pvals, size=nkl)
            tkl = np.argmax(draws, axis=1)
            noise_block = np.random.normal(0, 0.3, size=(nkl, long_))
            signal_block = fun[tkl, :] + noise_block
            block_data = signal_block.reshape((n_rows, n_cols, long_), order='F')
            X[np.ix_(rows_idx, cols_idx)] = block_data
            if bivariate:
                noise_block2 = np.random.normal(0, 0.3, size=(nkl, long_))
                signal_block2 = fun2[tkl, :] + noise_block2
                block_data2 = signal_block2.reshape((n_rows, n_cols, long_), order='F')
                X2[np.ix_(rows_idx, cols_idx)] = block_data2
            Y_block = tkl.reshape((n_rows, n_cols), order='F')
            Y[np.ix_(rows_idx, cols_idx)] = Y_block
    perm_rows = np.random.permutation(n)
    perm_cols = np.random.permutation(p)
    Z = Z[perm_rows]
    W = W[perm_cols]
    if bivariate:
        X = X[np.ix_(perm_rows, perm_cols)]
        X2 = X2[np.ix_(perm_rows, perm_cols)]
    else:
        X = X[np.ix_(perm_rows, perm_cols)]
    Y = Y[np.ix_(perm_rows, perm_cols)]
    if bivariate:
        return {"data1": X, "data2": X2, "row_clust": Z - 1, "col_clust": W - 1}
    else:
        return {"data": X, "row_clust": Z - 1, "col_clust": W - 1}
'''
lbm_simdata = lbm_sim_data()
'''


def sas_sim_data(scenario, n_i = 50, nbasis = 30, length_tot = 50, var_e = 1,
                 var_b = 1, seed = 123):
    grid = list(np.linspace(0, 1, length_tot))
    domain = [0, 1]
    X_basis = create_bspline_basis(domain, norder=4, nbasis=nbasis)
    mean_list = []
    
    # Generate data by scenario
    if scenario == 0:
        part = nbasis // 6
        mean1 = np.concatenate((np.repeat(1.5, part), np.repeat(0, nbasis - part)))
        mean2 = np.concatenate((np.repeat(-1.5, part), np.repeat(0, nbasis - part)))
        mean_list = [mean1, mean2]
        clus_true = np.repeat(np.arange(1, 3), n_i)
    elif scenario == 1:
        part = nbasis // 6
        mean1 = np.concatenate((np.repeat(3, part),
                                np.repeat(1.5, part),
                                np.repeat(0, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean2 = np.concatenate((np.repeat(0, part),
                                np.repeat(1.5, part),
                                np.repeat(0, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean3 = np.concatenate((np.repeat(0, part),
                                np.repeat(-1.5, part),
                                np.repeat(0, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean_list = [mean1, mean2, mean3]
        clus_true = np.repeat(np.arange(1, 4), n_i)
    elif scenario == 2:
        part = nbasis // 6
        mean1 = np.concatenate((np.repeat(1.5, part),
                                np.repeat(3, part),
                                np.repeat(1.5, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean2 = np.concatenate((np.repeat(1.5, part),
                                np.repeat(0, part),
                                np.repeat(1.5, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean3 = np.concatenate((np.repeat(-1.5, part),
                                np.repeat(0, part),
                                np.repeat(-1.5, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean4 = np.concatenate((np.repeat(-1.5, part),
                                np.repeat(-3, part),
                                np.repeat(-1.5, part),
                                np.repeat(0, nbasis - 3 * part)))
        mean_list = [mean1, mean2, mean3, mean4]
        clus_true = np.repeat(np.arange(1, 5), n_i)
    else:
        raise ValueError("Unknown scenario")
    mu_coef = np.transpose(np.vstack(mean_list))
    mu_fd = fd(mu_coef, X_basis)
    cov_mat = np.eye(nbasis) * var_b
    if len(mean_list) == 1:
        X_coef = np.transpose(multivariate_normal(mean_list[0], cov_mat).rvs(size=n_i))
    else:
        X_coef = np.transpose(multivariate_normal(mean_list[0], cov_mat).rvs(size=n_i))
        for ii in range(1, len(mean_list)):
            X_coef = np.concatenate((X_coef, np.transpose(multivariate_normal(mean_list[ii], cov_mat).rvs(size=n_i))), axis=1) 
    X_fd = fd(X_coef, X_basis)
    X = eval_fd(grid, X_fd)
    noise = np.random.normal(0, np.sqrt(var_e), size=X.shape)
    X = X + noise
    return {
        "X": X,
        "X_fd": X_fd,
        "mu_fd": mu_fd,
        "grid": grid,
        "clus": clus_true}
'''
sas_simdata = sas_sim_data(0, n_i = 20,var_e = 1,var_b = 0.25)
'''


def sparse_sim_data(n, x, paramC, plot = False):
    a = 3
    bpert = 0.5
    temp = a - 4 * (1 - x) * paramC / (1 - paramC)
    temp[x <= paramC] = (a - 4 * x)[x <= paramC]
    temp2 = np.full_like(x, bpert)
    temp2[x > paramC] = (bpert * (1 - x) / (1 - paramC))[x > paramC]
    fx = np.array([(np.random.normal(3, 0.5) * np.sin(np.random.normal(2, 0.25) * np.pi * x) + np.random.normal(3, 0.5)) * (np.random.normal(3, 0.5) - 4 * x) + np.random.normal(0, 0.5) for _ in range(n)]).T
    fx2 = np.array([(np.random.normal(3, 0.5) * np.sin(np.random.normal(2, 0.25) * np.pi * x) + np.random.normal(3, 0.5)) * temp + temp2 for _ in range(n)]).T
    data = np.hstack((fx, fx2))
    part_vera = np.concatenate((np.repeat(1, n), np.repeat(2, n)))
    if plot:
        plt.plot(x, data, linestyle='solid')
        plt.title("Set of synthetic data")
        plt.show()
    return {'data':data,
            'cluster':part_vera}
'''
sparse_simdata = sparse_sim_data(n = 10, x =  np.linspace(0, 1, 100), paramC = 2)
'''


def cvx_sim_data():
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    data_path = current_dir / 'simulation_data' / 'cvx_sim_data.csv'
    cvx_data = pd.read_csv(data_path)
    return cvx_data
'''
cvx_simdata = cvx_sim_data()
'''

def ssvd_sim_data():
    current_file_path = Path(__file__).resolve()
    current_dir = current_file_path.parent
    data_path = current_dir / 'simulation_data' / 'ssvd_sim_data.csv'
    ssvd_data = pd.read_csv(data_path).values
    return ssvd_data
'''
ssvd_simdata = ssvd_sim_data()
'''

