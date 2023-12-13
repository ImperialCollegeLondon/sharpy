import string, csv, string, os, numpy as np, sys, io, cProfile, pstats
from pstats import SortKey

def profile_analse(pr, settings, names = ['beam', 'uvlm']):

    prof_dir = settings['SHARPy']['route'] + settings['SHARPy']['case'] + '_profile.txt'
    sys.stdout = open(prof_dir, 'w')
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')            
    ps.print_stats()
    print(s.getvalue())
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    if names[-1] != "other":
        names.append("other")
    t_names = np.zeros(len(names), dtype=float)
    t_total = 0.0

    with open(prof_dir, newline='\n') as f: 
            csv_f = csv.reader(f, delimiter=' ', skipinitialspace=True)
            
            for row in csv_f:
                try:
                    t_total += float(list(row)[1])
                    for i in range(len(names)-1):
                        if list(row)[5].find(names[i]) != -1:
                            t_names[i] += float(list(row)[1])
                except: () 

    t_names[-1] = t_total - np.sum(t_names)

    print("\n")        
    for i in range(len(names)):
        print(names[i] + ' time = ' + f'{t_names[i]:.3f}' + ' seconds (' + f'{t_names[i]/t_total*100:.3f}' + '%)')
    print('\nTotal time = ' + f'{t_total:.3f}' + ' seconds\n')

    
    return {'names':names, 't_names':t_names, 't_total':t_total}