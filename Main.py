
import os
import json
import time
import numpy as np
from Train import Train

import warnings
warnings.filterwarnings("ignore")

def main():
    print("pid: {}".format(os.getpid()))
    print(time.strftime("*****%Y-%m-%d %H:%M:%S*****",time.localtime()))

    Config = json.load(open('config.json','r'))
    print(Config)

    results = {}   
    handle = Train(Config)
    val_loader = handle.val_loader

    for fold,loaders in handle.folds_loader.items():

        if not Config["models"][Config["active_model"]]["evaluate_only"]:
            print(f'[+]Start training {fold}: '+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            train_loader, test_loader, train_len, test_len = loaders['train'],loaders['test'],loaders['num'][0],loaders['num'][1]
            model,best_params = handle.train_HEARD(fold,train_loader,val_loader)
        
        print(f'[+]Start test {fold}:'+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
        model_dir = Config["models"][Config["active_model"]]["model_dir"]+Config["active_model"]+'/'+Config["models"][Config["active_model"]]["dataset"]+f'/{fold}'+'/best_model.m'
        acc,recall,precision,f1,er,sea = handle.test_HEARD(loaders['test'],model_dir)  
        results[fold] = {'test_acc':acc,'test_recall':recall,'test_precision':precision,'test_f1':f1,'test_er':er,'sea':sea}

    print("[+] Avg Results: Acc {:.4f}| R {:.4f} | P {:.4f} | F {:.4f} | ER {:.4f} | SEA {:.4f}".format(np.mean([results[x]['test_acc'] for x in results]),
                                                                                        np.mean([results[x]['test_recall'] for x in results]),
                                                                                        np.mean([results[x]['test_precision'] for x in results]),
                                                                                        np.mean([results[x]['test_f1'] for x in results]),
                                                                                        np.mean([results[x]['test_er'] for x in results]),
                                                                                        np.mean([results[x]['sea'] for x in results])))

if __name__ == '__main__':
    main()
    print(f'[+]Done: '+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
