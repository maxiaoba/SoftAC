import csv
import os.path
import matplotlib 
matplotlib.rcParams.update({'font.size': 10})
from matplotlib import pyplot as plt
import numpy as np

itr_interval = 100
max_itr = 2e4

fields = [
            'return-average',
            ]
itr_name = 'epoch'
min_loss = [-np.inf]
max_loss = [np.inf]
prepath = "./Data/Hopper"
plot_path = "./Data/Hopper"

policies = [
            'SAC_Gaussian',
            'SAC_LSP',
            'FlowQ_Gaussian',
            'FlowQ_LSP',
            'SAC_Gaussianlr0.0001',
            'SAC_LSPlr0.0001',
            'FlowQ_Gaussianlr0.0001',
            'FlowQ_LSPlr0.0001',
            'FlowQ_Gaussiancg10.0',
            'FlowQ_LSPcg10.0',
            'FlowQ_Gaussiancg100.0',
            'FlowQ_LSPcg100.0',
        ]
policy_names = policies
colors = []
for pid in range(len(policies)):
    colors.append('C'+str(pid))

extra_name = ''

pre_name = ''
post_name = ''

plot_name = extra_name

for fid,field in enumerate(fields):
    print(field)
    fig = plt.figure(fid,figsize=(10,10))
    legends = []
    plts = []
    for (policy_index,policy) in enumerate(policies):
        policy_path = pre_name+policy+post_name
        Itrs = []
        Losses = []
        min_itr = np.inf
        for trial in range(3):
            file_path = prepath+'/'+policy_path+'/'+'seed'+str(trial)+'/process.csv'
            if os.path.exists(file_path):
                print(policy+'_'+str(trial))
                itrs = []
                losses = []
                loss = []
                with open(file_path) as csv_file:
                    if '\0' in open(file_path).read():
                        print("you have null bytes in your input file")
                        csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                    else:
                        csv_reader = csv.reader(csv_file, delimiter=',')

                    for (i,row) in enumerate(csv_reader):
                        if i == 0:
                            entry_dict = {}
                            for index in range(len(row)):
                                entry_dict[row[index]] = index
                        else:
                            itr = i-1#int(float(row[entry_dict[itr_name]]))
                            if itr > max_itr:
                                break
                            loss.append(np.clip(float(row[entry_dict[field]]),
                                                min_loss[fid],max_loss[fid]))
                            if itr % itr_interval == 0:
                                itrs.append(itr)
                                loss = np.mean(loss)
                                losses.append(loss)
                                loss = []
                    if len(losses) < min_itr:
                        min_itr = len(losses)
            Losses.append(losses)
        Losses = [losses[:min_itr] for losses in Losses]
        itrs = itrs[:min_itr]
        Losses = np.array(Losses)
        print(Losses.shape)
        y = np.mean(Losses,0)
        yerr = np.std(Losses,0)
        plot, = plt.plot(itrs,y,colors[policy_index])
        plt.fill_between(itrs,y+yerr,y-yerr,linewidth=0,
                            facecolor=colors[policy_index],alpha=0.3)
        plts.append(plot)
        legends.append(policy_names[policy_index])

    plt.legend(plts,legends,loc='best')
    plt.xlabel('Itr')
    plt.ylabel(field) 
    fig.savefig(plot_path+'/'+plot_name+'_'+"_".join(field.split('/'))+'.pdf')
    plt.close(fig)