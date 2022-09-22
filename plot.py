#!/usr/bin/env python
# coding: utf-8

# In[31]:


from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import math, sys, re, statistics, os

from matplotlib.pyplot import figure

def group_data(lines: [str]) -> [str]:
    result = {}

    for line in lines:
        parts = line.split(',')
        if parts[0] in result:
            result[parts[0]].append((int(parts[1]), int(parts[2])))
        else:
            result[parts[0]] = [(int(parts[1]), int(parts[2]))]

    return result

def insert_bench(benches_map, bench_preset, bench_op, bench_limbs, time):
    if not bench_op in benches_map:
        benches_map[bench_op] = {}  
    if not bench_preset in benches_map[bench_op]:
        benches_map[bench_op][bench_preset] = {}
    if not bench_limbs in benches_map[bench_op][bench_preset]:
        benches_map[bench_op][bench_preset][bench_limbs] = {'data':[]}

    benches_map[bench_op][bench_preset][bench_limbs]['data'].append(time)

#input_lines = sys.stdin.readlines()
def read_go_arith_benchmarks(file_name):
    input_lines = []
    with open(file_name) as f:
        input_lines = f.readlines()

    benches_map = {}

    for line in input_lines[4:-2]:
        parts = [elem for elem in line[13:].split(' ') if elem and elem != '\t']

        bench_full = parts[0][:-2]
        #bench_full = re.search(r'(.*)?(#.*-.*$)', parts[0]).groups()[0]
        if '#' in parts[0] and parts[0].index('#'):
            bench_full = parts[0].split('#')[0]

        bench_preset = bench_full.split('_')[0]
        bench_op = bench_full.split('_')[1]
        bench_limbs = bench_full.split('_')[2]
        time = float(parts[2])
        unit = re.search(r'(.*)\/', parts[3]).groups()[0]

        if unit != 'ns':
            raise Exception("expected ns got {}".format(unit))

        insert_bench(benches_map, bench_preset, bench_op, bench_limbs, time)

    for bench_op in benches_map.keys():
        for bench_preset in benches_map[bench_op].keys():
            for bench_limbs in benches_map[bench_op][bench_preset].keys():
                item = benches_map[bench_op][bench_preset][bench_limbs]
                item['stddev'] = statistics.stdev(item['data'])
                item['mean'] = statistics.mean(item['data'])
                benches_map[bench_op][bench_preset][bench_limbs] = item # TODO is this necessary in python?
            #print("{},{},{},{},{}".format(bench_preset, bench_op, bench_limbs, item['mean'], item['stddev']))
    return benches_map

go_arith_benchmarks = read_go_arith_benchmarks("benchmarks-results/go-arith-benchmarks.txt")

def format_bench_data_for_graphing(x_range, bench_data, label, color):
    x_vals = []
    y_vals = []
    y_errs = []

    for limb_val in bench_data.keys():
        assert int(limb_val) % 64 == 0, "invalid limb bits"
        x_val = int(limb_val) // 64
        y_val = bench_data[limb_val]['mean']
        y_err = bench_data[limb_val]['stddev']
        x_vals.append(x_val)
        y_vals.append(y_val)
        y_errs.append(y_err)
    
    return (x_range, x_vals, y_vals, y_errs, color, label, 'o')

def scatterplot_ns_data(fname: str, name: str, annotate: bool, args):
    x_min_all = min([min([v for v in d[1]]) for d in args])
    x_max_all = max([max([v for v in d[1]]) for d in args])
    y_min_all = min([min([v for v in d[2]]) for d in args])
    y_max_all = max([max([v for v in d[2]]) for d in args])
    span_x = x_max_all - x_min_all
    span_y = y_max_all - y_min_all

    plt.rcParams["figure.figsize"] = (20, 10)
    fig, ax = plt.subplots()
    plt.ylim(0, y_min_all + span_y * 1.2)
    plt.xlim(0, x_min_all + span_x * 1.2)

    for (x_range, x_vals, y_vals, y_errs, color, label, marker) in args:
        assert len(x_vals) == len(y_vals)
        assert len(y_vals) == len(y_errs)

        plt.xlabel("number of limbs")
        plt.ylabel("runtime (ns)")
        
        print(x_vals)
        for x, y, y_err in zip(x_vals, y_vals, y_errs):
            if x < x_range[0] or x > x_range[1]:
                continue

            if annotate:
                ax.annotate(y, (float(x) + 0.2, float(y)))
            
            ax.errorbar(x=x, y=y, xerr=0.0, yerr=y_err, fmt='o', color=color)

       # ax.plot(x_vals, y_vals, marker, color=color, label=label)
    plt.legend(loc="upper left")
    ax.set(title=name)
    plt.savefig(fname)

def fit_quadratic(dataset):
    xs = [val[0] for val in dataset]
    ys = [val[1] for val in dataset]

    eqn = np.polyfit(np.array(xs), np.array(ys), 2)
    eqn = [round(val, 2) for val in eqn]
    lof_y = [x ** 2 * eqn[0] + x * eqn[1] + eqn[2] for x in xs]
    return eqn, list(zip(xs, lof_y))

def fit_linear(dataset):
    xs = [val[0] for val in dataset]
    ys = [val[1] for val in dataset]

    eqn = np.polyfit(np.array(xs), np.array(ys), 1)
    eqn = [round(val, 2) for val in eqn]
    
    # make sure the line of fit goes thru the first limb
    new_y_intercept_addmod = dataset[0][1] - abs(abs(eqn[0]) - abs(eqn[1]))
    print(new_y_intercept_addmod)
    eqn[1] = new_y_intercept_addmod

    lof_y = [x * eqn[0] + eqn[1] for x in xs]
    return eqn, list(zip(xs, lof_y))

def stitch_data(data1, data2, cutoff: int):
    result = [] 
    print("data1 is {}".format(data1))
    x_values = [[x_val for x_val in data1[key]] for key in data1.keys()]
    print(x_values)
    for i in range(1, cutoff):
        result.append((data1['x_values'][str(i * 64)], data1['y_value'][str(i * 64)]))
    for i in range(cutoff, 65):
        result.append((data2['x_values'][str(i * 64)], data2['y_value'][str(i * 64)]))

    return result

def stitch_gas_model(model1, model2, cutoff: int):
    pass

fast_mulmont_limb_cutoff = 49
mulmont_benches = go_arith_benchmarks['mulmont']
#eqn_mulmont_lof, mulmont_lof = fit_quadratic(zip([mulmont_benches['y_vals'])[:fast_mulmont_cutoff]
mulmont_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['mulmont']['non-unrolled'], 'mulmont', 'red')
mulmont_generic_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['mulmont']['generic'], 'mulmont-generic', 'blue')
setmod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['setmod']['non-unrolled'], 'setmod-non-unrolled', 'red')
setmod_generic_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['setmod']['generic'], 'setmod-generic', 'blue')
addmod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['addmod']['non-unrolled'], 'addmod-non-unrolled', 'red')
submod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['submod']['non-unrolled'], 'submod-non-unrolled', 'red')

scatterplot_ns_data("charts/mulmont_generic.png", "MULMONTMAX Generic Benchmarks", False, [mulmont_non_unrolled_data, mulmont_generic_data])
scatterplot_ns_data('charts/setmod.png', 'SETMOD Benchmarks', False, [setmod_non_unrolled_data, setmod_generic_data])
scatterplot_ns_data('charts/addmod.png', 'ADDMOD Benchmarks', False, [addmod_non_unrolled_data])
scatterplot_ns_data('charts/submod.png', 'SUBMOD Benchmarks', False, [submod_non_unrolled_data])

sys.exit(0)

stitch_data(go_arith_benchmarks['mulmont']['generic'], go_arith_benchmarks['mulmont']['non-unrolled'], 49)
