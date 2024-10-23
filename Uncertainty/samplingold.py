import configparser
import os

import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader
from pathos import multiprocessing, pools
from argparse import ArgumentParser, RawTextHelpFormatter
import datetime as dt
from uncertainty import uncertaintySupervisor, extractQoI, deorbitBurn
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from TITAN import load_and_run_cfg as runTITAN
import shutil
from distutils.dir_util import copy_tree
import numpy as np

from matplotlib import pyplot as plt
from messaging import messenger

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def wrapper(cfg):
    runTITAN(cfg,'')
    outputs = extractQoI(cfg, cfg['Options']['Output_folder']+'/Data/data.csv')
    return outputs

def createOracle(cfg):
    print('Generating Oracle...')
    if os.path.exists('TempOracle'): shutil.rmtree('TempOracle')
    os.mkdir('TempOracle')

    oracle_cfg = configparser.ConfigParser()
    oracle_cfg.read_dict(cfg)
    oracle_cfg.set('Options', 'Output_folder', 'TempOracle')
    oracle_cfg.set('Options','Load_mesh','False')
    oracle_cfg.set('Options','Fidelity','Low')
    oracle_cfg.set('QoI','Outputs','Mass')

    with open(args.yamlfile,'r') as file: uncertainties = yaml.load(file,Loader)
    if 'deorbit' in uncertainties:
        burndata = uncertainties['deorbit'].copy()
        burndata['sigma_pointing_fixed'] = 0
        burndata['sigma_pointing_proportional'] = 0
        burndata['sigma_magnitude_fixed'] = 0
        burndata['sigma_magnitude_proportional'] = 0
        deorbitBurn(oracle_cfg,burndata)


    if not cfg.getboolean('GRAM', 'Uncertain'):
        print('...pregenerating mesh')
        oracle_cfg.set('Options', 'num_iters', '1')
    else:
        print('...uncertain GRAM selected, running full sim')
        oracle_cfg.set('GRAM','Uncertain','False')


    runTITAN(oracle_cfg,'')
    mass = extractQoI(oracle_cfg,'TempOracle/Data/data.csv')[0]


    if 'deorbit' in uncertainties:
        uncertainties['deorbit']['vehicle_mass']=float(mass)

    with open(args.yamlfile, 'w') as file: yaml.dump(uncertainties,file)




def loadOracleData(directory):
    subfolders =['Data','GRAM','Restart']

    if os.path.exists(directory): shutil.rmtree(directory)

    os.mkdir(directory)
    [os.mkdir(directory + '/' + sub) for sub in subfolders]
    [copy_tree(args.oracle + '/' + sub, directory + '/' + sub) for sub in subfolders]


def run_batch(n_processes,cfg):

    pool = pools.ProcessPool(processes=n_processes)
    cfgs=[]
    for i in range(n_processes):
        new_cfg = configparser.ConfigParser()
        new_cfg.read_dict(cfg)
        cfgs.append(UQ.sampleConfig(new_cfg, (i % n_processes)))
        loadOracleData(cfgs[i]['Options']['Output_folder'])


    outputs = pool.map(wrapper,cfgs)
    outputs = np.reshape(outputs, [n_processes, -1])

    return outputs


if __name__ == "__main__":
    # To run TITAN, it requires the user to specify a configuration
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("-N", "--num",
                        dest="n_processes",
                        type=int,
                        help="number of processes to run, leaving unspecified will attempt to use all available processes",
                        metavar="n")
    parser.add_argument("-c", "--config",
                        dest="cfgfile",
                        type=str,
                        help="filename of base config file",
                        metavar="cfg")
    parser.add_argument("-u", "--uqfile",
                        dest="yamlfile",
                        type=str,
                        help="filename of .yaml uncertainty file",
                        metavar="yaml")
    parser.add_argument("-O", "--oracle",
                        dest="oracle",
                        type=str,
                        help="path to \"Oracle\" folder containing prior TITAN run, necessary to run uncertain GRAM,if needed one will be generated when not specified",
                        metavar="path")
    parser.add_argument("-p", "--print",
                        dest="show_prints",
                        type=bool,
                        help="bool to toggle whether TITAN processes print text to terminal",
                        metavar="y/N")

    args=parser.parse_args()

    if not 'TITAN' in os.getcwd()[-10:]:
        os.chdir('../')


    n_procs = multiprocessing.cpu_count() if not args.n_processes else args.n_processes

    msg =messenger(threshold=300)

    if not args.cfgfile: raise Exception('The user needs to provide a file!.\n')

    cfg = configparser.ConfigParser()
    cfg.read(args.cfgfile)

    if not args.oracle:
        createOracle(cfg)
        args.oracle='TempOracle'
    cfg.set('Options','Load_mesh','True')

    UQ = uncertaintySupervisor(isActive=True,rngSeed=dt.datetime.now().microsecond)

    UQ.constructInputs(args.yamlfile)


    outputs = []
    n_samples = cfg.getint('Options','Num_runs')
    n_batches = n_samples // n_procs
    n_remainder = n_samples % n_procs

    print('Beginning sampling run of ',n_batches,' batches across ',n_procs,' processes with ',n_remainder,' remainder runs afterward (',n_batches*n_procs+n_remainder,' total samples)')

    for i in range(n_batches):

        if not args.show_prints:
            with HiddenPrints(): output=run_batch(n_procs,cfg)
        else: output=run_batch(n_procs,cfg)
        if len(outputs) < 1: outputs=output
        else: outputs=np.vstack((outputs,output))
        msg.print_n_send('Batch '+str(i+1)+'/'+str(n_batches)+' completed')

    if n_remainder:
        if not args.show_prints:
            with HiddenPrints(): output=run_batch(n_remainder,cfg)
        else: output=run_batch(n_remainder,cfg)

        outputs = np.vstack((outputs, output)) if len(outputs)>0 else output


    plt.style.use('dark_background')
    fig1, ax1 = plt.subplots(ncols=1,sharex=True)
    ax1.scatter(x = outputs[:,1], y =outputs [:,0],marker='x')
    ax1.set_aspect('equal')
    ax1.set_xlabel('Longitude (deg)')
    ax1.set_ylabel('Latitude (deg)')
    plt.show()


