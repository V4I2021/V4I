import pandas as pd
import scipy
from scipy.stats import pearsonr

carSales = "D:\\arslanaWu\\组会\\data_facts\\data set\\CarSales.csv"

data_list = pd.read_csv(carSales, header=0)

column_list = data_list.head(0).columns.values.tolist()

value_list = dict()
for i in range(len(column_list) - 1):
    value_list[column_list[i]] = list(data_list[column_list[i]].unique())
# print(value_list)

break_down = 'Year'
break_down_value_list = list(data_list[break_down].unique())
# print(break_down_value_list)

subspace_col_list = [x for x in column_list[0:3] if (x not in break_down)]
# print(subspace_col_list)

res_of_breakdown = []
label_list = []
# print('no subspace, breakdown is {}'.format(break_down))
sum_list = []
for break_down_value in break_down_value_list:
    # sum value of breakdown = breakdown value
    x = data_list.loc[(data_list[break_down] == break_down_value)][
        column_list[-1]].sum()
    sum_list.append(x)
# print(sum_list)
res_of_breakdown.append(sum_list)
label_list.append('*;*;')

for col in subspace_col_list:
    value_of_col = value_list[col]
    # print(value_of_col)
    for value in value_of_col:
        # print('subspace of {} = {}, breakdown is {}'.format(col, value, break_down))
        sum_list = []
        for break_down_value in break_down_value_list:
            # sum value of breakdown = breakdown value
            x = data_list.loc[(data_list[col] == value) & (data_list[break_down] == break_down_value)][
                column_list[-1]].sum()
            sum_list.append(x)
        # print(sum_list)
        res_of_breakdown.append(sum_list)
        if col == 'Brand':
            label_list.append('{};*;'.format(value))
        else:
            label_list.append('*;{};'.format(value))
        # label_list.append('subspace of {} = {}'.format(col, value))

for value1 in value_list[subspace_col_list[0]]:
    for value2 in value_list[subspace_col_list[1]]:
        # print('subspace of {} = {} and {} = {}, breakdown is {}'.format(subspace_col_list[0], value1,
        #                                                                 subspace_col_list[1], value2,
        #                                                                 break_down))
        sum_list = []
        for break_down_value in break_down_value_list:
            x = data_list.loc[(data_list[subspace_col_list[0]] == value1)
                              & (data_list[subspace_col_list[1]] == value2)
                              & (data_list[break_down] == break_down_value)][
                column_list[-1]].sum()
            sum_list.append(x)
        res_of_breakdown.append(sum_list)
        label_list.append('{};{};'.format(value1, value2))
        # label_list.append('subspace of {} = {} and {} = {}'.format(subspace_col_list[0], value1,
        #                                                            subspace_col_list[1], value2,
        #                                                            ))

f = open("D:\\arslanaWu\\组会\\data_facts\\result\\GetCorrelationRes_CarSales1.csv", 'w')
print("subspace1,subspace2,breakdown,correlation")
f.write("subspace1,subspace2,breakdown,correlation")
f.write('\n')
for i in range(0, len(res_of_breakdown)):
    for j in range(i + 1, len(res_of_breakdown)):
        # print("{}, {}".format(res_of_breakdown[i], res_of_breakdown[j]))
        corr, _ = pearsonr(res_of_breakdown[i], res_of_breakdown[j])
        # p = scipy.stats.norm(0, 0.05).pdf(corr)
        print("{},{},{},{}".format(label_list[i], label_list[j], 'Year', corr))
        f.write("{},{},{},{}".format(label_list[i], label_list[j], 'Year', corr))
        f.write('\n')
f.close()

# for break_down in itertools.combinations(column_list[0:3], 1):
#     res_of_breakdown = []
#
#     # print(break_down)
#     break_down_value_list = list(data_list[break_down[0]].unique())
#     # print(break_down_value_list)
#
#     subspace_col_list = [x for x in column_list[0:3] if (x not in break_down)]
#     # print(subspace_col_list)
#
#     for col in subspace_col_list:
#         value_of_col = value_list[col]
#         # print(value_of_col)
#         for value in value_of_col:
#             print('subspace of {} = {}, breakdown is {}'.format(col, value, break_down[0]))
#             sum_list = []
#             for break_down_value in break_down_value_list:
#                 # sum value of breakdown = breakdown value
#                 x = data_list.loc[(data_list[col] == value) & (data_list[break_down[0]] == break_down_value)][
#                     column_list[-1]].sum()
#                 sum_list.append(x)
#             # print(sum_list)
#             res_of_breakdown.append(sum_list)

# for break_down in itertools.combinations(column_list[0:3], 2):
#     res_of_breakdown = []
#
#     # print(break_down)
#     break_down_value_list = np.array(
#         data_list.groupby(list(break_down)).size().reset_index(name='Freq')).tolist()
#     # print(break_down_value_list)
#
#     subspace_col = [x for x in column_list[0:3] if (x not in break_down)]
#     # print(subspace_col)
#
#     for col in subspace_col:
#         value_of_col = value_list[col]
#         for value in value_of_col:
#             print('subspace of {} = {}, breakdown is {}'.format(col, value, break_down))
#             sum_list = []
#             for break_down_value in break_down_value_list:
#                 x = data_list.loc[(data_list[col] == value)
#                                   & (data_list[break_down[0]] == break_down_value[0])
#                                   & (data_list[break_down[1]] == break_down_value[1])][
#                     column_list[-1]].sum()
#                 sum_list.append(x)
#             res_of_breakdown.append(sum_list)
