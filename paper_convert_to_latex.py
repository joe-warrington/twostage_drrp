# (c) 2017-2019 ETH Zurich, Automatic Control Lab, Joe Warrington, Dominik Ruchti

import pandas as pd

suffix_list = ['_det2s_corrected', '_regular', '_halfinteger', '_integer', '_random']

for suffix in suffix_list:

    rdf = pd.read_csv('output/mean_std_stats' + suffix + '.csv', index_col=[0, 1, 2])

    cost_mean_list = rdf['Cost_mean'].tolist()
    nr = len(cost_mean_list)
    cost_std_list = rdf['Cost_std'].tolist()
    cost_mean_std_list = ["%.2f +- %.2f" % (cost_mean_list[i], cost_std_list[i]) for i in range(nr)]
    rdf['Cost_mean_std'] = cost_mean_std_list
    sr_mean_list = rdf['SR_mean'].tolist()
    sr_std_list = rdf['SR_std'].tolist()
    sr_mean_std_list = ["%.2f +- %.2f" % (sr_mean_list[i], sr_std_list[i]) for i in range(nr)]
    rdf['SR_mean_std'] = sr_mean_std_list
    cost_rel_mean_list = rdf['Cost_rel_mean'].tolist()
    cost_rel_std_list = rdf['Cost_rel_std'].tolist()
    cost_rel_mean_std_list = ["%.2f +- %.2f" % (cost_rel_mean_list[i],
                                                cost_rel_std_list[i]) for i in range(nr)]
    rdf['Cost_rel_mean_std'] = cost_rel_mean_std_list
    sr_rel_mean_list = rdf['SR_rel_mean'].tolist()
    sr_rel_std_list = rdf['SR_rel_std'].tolist()
    sr_rel_mean_std_list = ["%.2f +- %.2f" % (sr_rel_mean_list[i],
                                              sr_rel_std_list[i]) for i in range(nr)]
    rdf['SR_rel_mean_std'] = sr_rel_mean_std_list

    cost_unstack = rdf['Cost_mean_std'].unstack(2)
    sr_unstack = rdf['SR_mean_std'].unstack(2)
    cost_rel_unstack = rdf['Cost_rel_mean_std'].unstack(2)
    sr_rel_unstack = rdf['SR_rel_mean_std'].unstack(2)
    k_list = [0, 200, 201] if suffix == '_random' else [0, 50, 51]
    print "Initial service rate, " + suffix + ":"
    print sr_unstack[k_list[0]].to_latex().replace("+-", "$\pm$")
    print "Initial cost, " + suffix + ":"
    print cost_unstack[k_list[0]].to_latex().replace("+-", "$\pm$")
    print "Service rate rel, " + suffix + ":"
    print sr_rel_unstack[k_list[1:]].to_latex().replace("+-", "$\pm$")
    print "Cost rel, " + suffix + ":"
    print cost_rel_unstack[k_list[1:]].to_latex().replace("+-", "$\pm$")

for i, filename in enumerate(['output/iter_times' + suffix + '.csv' for suffix in suffix_list]):
    if i == 0:
        df = pd.read_csv(filename, index_col=[0, 1])[['t1_mean']]
        df2 = pd.read_csv(filename, index_col=[0, 1])[['t2_mean']]
    elif i in [1, 2, 3, 4]:
        df = df.join(other=pd.read_csv(filename, index_col=[0, 1])[['t1_mean']], how='outer',
                     lsuffix='', rsuffix=suffix_list[i])
        df2 = df2.join(other=pd.read_csv(filename, index_col=[0, 1])[['t2_mean']], how='outer',
                       lsuffix='', rsuffix=suffix_list[i])
print df.round(3).to_latex()
print df2.round(5).to_latex()
