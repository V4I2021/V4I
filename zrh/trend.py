import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from scipy.stats import logistic
import matplotlib.pyplot as plt
import statistics
from scipy import optimize
import csv
import numpy as np
from scipy.stats import norm

dim_range = dict()
year_col = "Year"
# 需要修改year_col!!!!!!
insight_type = "Trend"
str_col = []
# str_col是subspace的列
full_str = []
# full_str是subspace的列 和 year_col
num_col = []
# measure
out_path = ""
measure = "Price"
# 其实和num_col是一样的...

'''
这个文件可以输出trend或者point的impact和significant结果
用 insight_type = "Trend"
insight_type = "Point"
切换
'''

def load_data(file_name, insight_type):
    global str_col
    global num_col
    global out_path
    global full_str
    in_path = "./Data/%s.csv" % file_name
    out_path = "./res/insight_%s_temp.csv" % file_name

    df = pd.read_csv(in_path)
    df = df.fillna(0)
#     num_col = df.select_dtypes(include=np.number).columns.tolist()

    '''
    直接人为给定了num_col, full_col, str_col
    '''
    num_col = [measure]
    str_col = ['Brand', 'Body', 'Year']
    full_str = ['Brand', 'Body', 'Year']
    
#     full_str = [i for i in df.columns.tolist() if i not in num_col]
    '''
    在str_col中删除year_col
    '''
    if insight_type == "Trend":
        str_col = [i for i in df.columns.tolist() if i in full_str and i != year_col]
    else:
        str_col = [i for i in df.columns.tolist() if i in full_str]
    # df[year_col] = [i[0:4] for i in df[year_col]]
    get_dim_range(df)

    print("String Column: ", str_col)
    print("Numeric Column: ", num_col)
    return df


def get_dim_range(dfm):
    '''
    获得subspace range
    '''
    for dim in full_str:
        dim_range[dim] = []
        dim_range[dim].extend(dfm[dim].unique())
    if insight_type == "Trend":
        dim_range[year_col] = []
        dim_range[year_col].extend(dfm[year_col].unique())
    print(dim_range)


def point_power_law(phi):
    '''
    power law 计算
    '''
    ordered_phi = {k: v for k, v in sorted(phi.items(), key=lambda item: item[1], reverse=True)}
    keys = list(ordered_phi.keys())
    values = list(ordered_phi.values())
    max_value = max(values)
    ydata = []
    for i in values:
        if i != max_value and i not in ydata:
            ydata.append(i)
    xdata = range(2, len(ydata) + 2)
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    pinit = [1.0, -1.0]

    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    powerLawFunc = lambda amp, index, x: amp * (x ** index)

    try:
        out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

        pfinal = out[0]
        covar = out[1]

        index = pfinal[1]
        amp = 10.0 ** pfinal[0]

        indexErr = np.sqrt(covar[1][1])
        ampErr = np.sqrt(covar[0][0]) * amp

        return [keys[0], values[0], powerLawFunc(amp, index, 1)]

    except TypeError:
        return [keys[0], values[0], -1]


def write_prediction_error(subspace, breakdown, bdval, val, predictedVal, err):
    
    subspace_str = []
    for dim in str_col:
        exist = False
        for space in subspace:
            if space in dim_range[dim]:
                exist = True
                subspace_str.append(space)
        if not exist:
            subspace_str.append("*")
    return (";".join(str(v) for v in subspace_str) + "," + str(breakdown) + "," + str(bdval) + "," + str(
        val) + "," + str(predictedVal) + "," + str(err))


def iterate_subspace(df):
    store_dfs = list()
    subspace_list = []
    breakdown_list = []

    for subspaceLen in range(0, len(str_col) + 1):
        for subspace_cols in itertools.combinations(str_col, subspaceLen):
            #       dims: subspace中的元素

            if insight_type == "Trend":
                breakdowns = [year_col]
            else:
                breakdowns = [x for x in str_col if (x not in subspace_cols)]

            subspace_elements = []
            for subspace_col in subspace_cols:
                subspace_elements.append(dim_range[subspace_col])
            subspaces = list(itertools.product(*subspace_elements))
            for subspace in subspaces:
                for breakdown in breakdowns:
                    for measure in num_col:
                        if len(subspace_cols) > 0:
                            dff = df.loc[
                                np.all([df[subspace_cols[i]] == subspace[i] for i in range(0, subspaceLen)], axis=0)]
                        else:
                            dff = df
                        df_group = dff.groupby(breakdown, as_index=False).sum()
                        if not df_group.empty:
                            store_dfs.append(df_group)
                            subspace_list.append(subspace)
                            breakdown_list.append(breakdown)

    if insight_type == "Trend":
        return trend_cal(store_dfs, breakdown_list, subspace_list)
    else:
        return point_cal(store_dfs, breakdown_list, subspace_list)


def trend_cal(store_dfs, breakdown_list, subspace_list):
    '''
    trend impact and significance calculation
    '''
    sig_scores = []
    r_sqs = []
    slopes = []
    impacts = []
    print(store_dfs[1])
    for idx, df_group in enumerate(store_dfs):
        impacts.append(df_group[measure].sum())
        x = df_group[year_col].values
        y = df_group[measure].values
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        slope = reg.coef_[0]
        slopes.append(slope[0])
        r_sq = reg.score(x, y)
        r_sqs.append(r_sq)

    mean_val = statistics.mean(slopes)
    std = statistics.stdev(slopes)

    for idx, df_group in enumerate(store_dfs):
        slope = slopes[idx]
        p = (1 - logistic.cdf(mean_val + abs(mean_val - slope), loc=mean_val, scale=std)) * 2
        res = r_sqs[idx] * (1 - p)
        sig_scores.append(res)

    return subspace_list, breakdown_list, sig_scores, impacts, [0 for i in range(0, len(impacts))]


def point_cal(store_dfs, breakdown_list, subspace_list):
    err_values = []
    subspace_remain = []
    breakdown_remain = []
    impacts = []
    breakdown_values = []

    for idx, df_group in enumerate(store_dfs):
        phi = dict(zip(df_group[breakdown_list[idx]], df_group[measure]))
        # print(phi)

        res = point_power_law(phi)
        if res == -1:
            pass
        else:
            err_values.append(res[1] - res[2])
            breakdown_values.append(res[0])
            subspace_remain.append(subspace_list[idx])
            breakdown_remain.append(breakdown_list[idx])
            impacts.append(df_group[measure].sum())

    np_error_values = np.array(err_values)
    std = np.std(np_error_values)
    mean = np.mean(np_error_values)
    print("Standard Deviation: ", std)
    print("Mean: ", mean)

    plt.figure(figsize=(10, 6))
    mu, std = norm.fit(np_error_values)
    plt.hist(np_error_values, bins=25, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.show()

    prob_res = norm.cdf(np_error_values, mu, std)
    sig_scores = [1 - i for i in prob_res]

    return subspace_remain, breakdown_remain, sig_scores, impacts, breakdown_values


def format_subspace(subspace):
    subspace_string = []
    for dim in full_str:
        exist = False
        for space in subspace:
            if space in dim_range[dim]:
                exist = True
                subspace_string.append(space)
        if not exist:
            subspace_string.append("*")
    return subspace_string


if __name__ == '__main__':
    '''
    set dataset
    '''
    df = load_data("carSales2", insight_type)
    
    
    '''
    计算impact(只有分子，分母还没算) and sig
    '''
    subspace_list, breakdown_list, sig_scores, impacts, breakdown_values = iterate_subspace(df)
    
    '''
    获得impact计算需要的分母
    '''
    sumImpact = df[num_col[0]].sum()
    # with open(out_path, 'w+', newline='') as csvfile:
    
    '''
    写入csv
    '''
    with open(out_path, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        
        '''
        写入标题
        '''
        spamwriter.writerow(['Subspace', 'Breakdown', 'Sig', 'Impact', 'Type', 'Breakdown_value'])
        
        '''
        写入数据
        '''
        for idx, val in enumerate(subspace_list):
            subspace_string = format_subspace(subspace_list[idx])
            spamwriter.writerow([';'.join(str(v) for v in subspace_string), str(breakdown_list[idx]), str(sig_scores[idx]),
                                 impacts[idx]/sumImpact, insight_type, breakdown_values[idx]])
