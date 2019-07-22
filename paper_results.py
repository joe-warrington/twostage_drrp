import pandas as pd
import numpy as np
import os, platform
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '' and platform.system() == 'Linux':
    mpl.use('Agg')  # Use non-interactive Agg backend on Linux server
from matplotlib import pyplot as plt
import os
mpl.rcParams['font.family'] = 'serif'

for suffix in ['_regular', '_integer', '_halfinteger', '_det2s_corrected', '_random']:
    # Import and concatenate results sheets
    suffix_len = len(suffix)
    filelist = [fname for fname in os.listdir('output')
                if fname[:5] == 'stats' and fname[-(suffix_len + 4):] == suffix + '.csv']
    dfl = []
    for f in filelist:
        df = pd.read_csv('output/' + f)
        try:
            df['SR_rel'] = df['SR'] - df.loc[(df['k'] == 0) | (str(df['k']) == '0.0')]['SR'][0]
            df['Cost_rel'] = df['Cost'] - df.loc[(df['k'] == 0) | (str(df['k']) == '0.0')]['Cost'][0]
        except Exception as e:
            print 'output/' + f
            df.head()
            print df.loc[(df['k'] == 0) | (str(df['k']) == '0.0')]
            print "df.loc[(dfl['k'] == 0) | (str(df['k']) == '0.0')]['SR'][0]", \
                df.loc[(df['k'] == 0) | (str(df['k']) == '0.0')]['SR'][0]
            print "df.loc[(dfl['k'] == 0) | (str(df['k']) == '0.0')]['Cost'][0]", \
                df.loc[(df['k'] == 0) | (str(df['k']) == '0.0')]['Cost'][0]
            raise SystemExit()
        dfl.append(df)
    dc = pd.concat(dfl, ignore_index=True)
    print "Suffix " + suffix + ": %d files found, containing %d records." % (len(filelist), len(dc))

    # Average results across network sizes, on the "Inst." column.
    iter_times = dc.loc[dc['Integer'] == '0.0'].groupby(['N', 'V'])['t1', 't2'].mean()
    iter_times_std = dc.loc[dc['Integer'] == '0.0'].groupby(['N', 'V'])['t1', 't2'].std()
    rec_count = dc.loc[(dc['Integer'] == '0.0') & (dc['k'] == 50)].groupby(['N', 'V'])['t1'].count()
    print rec_count
    try:
        iter_csv = iter_times.merge(right=iter_times_std, on=['N', 'V'], suffixes=('_mean', '_std'))
        print iter_csv.head()
    except KeyError as e:
        print iter_times
        print iter_times_std
        print e
        raise SystemExit()
    # iter_csv['N'] = iter_csv['N'].map(lambda x: "%d" % x)
    # iter_csv['V'] = iter_csv['V'].map(lambda x: "%d" % x)
    iter_csv.round(6).to_csv('output/iter_times' + suffix + '.csv')

    grouped = dc.groupby(['N', 'V', 'k'])['Cost', 'SR', 'Cost_rel', 'SR_rel']
    cost_sr, std_cost_sr = grouped.mean().dropna(), grouped.std().dropna()

    mean_std_df = cost_sr.merge(right=std_cost_sr, on=['N', 'V', 'k'], suffixes=('_mean', '_std'))
    mean_std_df.round(6).to_csv('output/mean_std_stats' + suffix + '.csv')

    save_plots = False
    if save_plots:
        for N, sub_df in mean_std_df.groupby(level=0):
            for V, sub_sub_df in sub_df.groupby(level=1):
                try:
                    k_list = [np.rint(m_idx[2]) for m_idx in sub_sub_df.index.values.tolist()]
                    sr_mean_list = sub_sub_df['SR_rel_mean'].tolist()
                    sr_std_list = sub_sub_df['SR_rel_std'].tolist()
                    cost_mean_list = sub_sub_df['Cost_rel_mean'].tolist()
                    cost_std_list = sub_sub_df['Cost_rel_std'].tolist()
                    if suffix == '_integer':
                        k_list = k_list[:-1]  # Correct for the fact that the integer sol's have an unneeded iteration 51.
                        sr_mean_list = sr_mean_list[:-1]
                        sr_std_list = sr_std_list[:-1]
                        cost_mean_list = cost_mean_list[:-1]
                        cost_std_list = cost_std_list[:-1]
                    plt.figure(figsize=(8, 4))
                    plt.plot(k_list, sr_mean_list, 'k')
                    plt.plot(k_list, np.array(sr_mean_list) + np.array(sr_std_list), 'k--')
                    plt.plot(k_list, np.array(sr_mean_list) - np.array(sr_std_list), 'k--')
                    plt.xlim([np.min(k_list), np.max(k_list)])
                    plt.xlabel('Iteration $k$')
                    plt.ylabel('Service rate increase (%)')
                    plt.title('%d nodes, %d RV' % (N, V) + ('' if V == 1 else 's'))
                    filename = 'output/sr_stats_%d_%d' % (N, V) + suffix + '.pdf'
                    plt.tight_layout()
                    print "Saving " + filename + "..."
                    plt.savefig(filename)
                    plt.close()
                    plt.figure(figsize=(8, 4))
                    plt.plot(k_list, cost_mean_list, 'k')
                    plt.plot(k_list, np.array(cost_mean_list) + np.array(cost_std_list), 'k--')
                    plt.plot(k_list, np.array(cost_mean_list) - np.array(cost_std_list), 'k--')
                    plt.xlim([np.min(k_list), np.max(k_list)])
                    # plt.ylim([0, cost_mean_list[-1] * 10])
                    plt.xlabel('Iteration $k$')
                    plt.ylabel('Cost change')
                    plt.title('%d nodes, %d RV' % (N, V) + ('' if V == 1 else 's'))
                    filename = 'output/cost_stats_%d_%d' % (N, V) + suffix + '.pdf'
                    plt.tight_layout()
                    print "Saving " + filename + "..."
                    plt.savefig(filename)
                    plt.close()
                except Exception as e:
                    print "Failed to generate graph 'output/sr_stats_%d_%d" % (N, V) + suffix + ".pdf'"
                    print "                  and/or 'output/cost_stats_%d_%d" % (N, V) + suffix + ".pdf'"
                    print e
