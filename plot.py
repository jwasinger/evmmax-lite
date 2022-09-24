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
        bench_limbs = int(bench_full.split('_')[2])
        assert bench_limbs % 64 == 0, "invalid bench limb bit width"
        bench_limbs //= 64
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

    for i, limb_val in enumerate(bench_data.keys()):
        if i + 1 < x_range[0] or i + 1 > x_range[1]:
            continue

        x_val = limb_val
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

        plt.xlabel("number of limbs")
        plt.ylabel("runtime (ns)")
        
        if len(y_errs) != 0:
            assert len(y_vals) == len(y_errs)
            for x, y, y_err in zip(x_vals, y_vals, y_errs):
                if x < x_range[0] or x > x_range[1]:
                    continue

                if annotate:
                    ax.annotate(y, (float(x) + 0.2, float(y)))
                
                ax.errorbar(x=x, y=y, xerr=0.0, yerr=y_err, fmt='o', color=color)
        else:
           ax.plot(x_vals, y_vals, marker, color=color, label=label)
    plt.legend(loc="upper left")
    ax.set(title=name)
    plt.savefig(fname)

def fit_quadratic(x_range, graphing_dataset):
    xs = []
    ys = []

    for i, x_val in enumerate(graphing_dataset[1]):
        if x_val < x_range[0] or x_val > x_range[1]:
            continue
        xs.append(x_val)
        ys.append(graphing_dataset[2][i])

    eqn = np.polyfit(np.array(xs), np.array(ys), 2)
    eqn = [round(val, 2) for val in eqn]
    lof_y = [x ** 2 * eqn[0] + x * eqn[1] + eqn[2] for x in xs]
    return eqn, list(zip(xs, lof_y))

def fit_linear(x_range, graphing_dataset):
    xs = []
    ys = []

    for i, x_val in enumerate(graphing_dataset[1]):
        if x_val < x_range[0] or x_val >= x_range[1]:
            continue
        xs.append(x_val)
        ys.append(graphing_dataset[2][i])

    eqn = np.polyfit(np.array(xs), np.array(ys), 1)
    eqn = [round(val, 2) for val in eqn]
    
    # make sure the line of fit goes thru the first limb
    # TODO make this optional (I think addmod/submod needed it to get proper models)
    # new_y_intercept_addmod = graphing_dataset[0][1] - abs(abs(eqn[0]) - abs(eqn[1]))
    # eqn[1] = new_y_intercept_addmod

    lof_y = [x * eqn[0] + eqn[1] for x in xs]
    return eqn, list(zip(xs, lof_y))

def stitch_model(model1, model2, cutoff: int):
    result = []
    for i, (x_val, y_val) in enumerate(model1):
        if i >= cutoff:
            break

        result.append((x_val, y_val))

    for i, (x_val, y_val) in enumerate(model2):
        result.append((x_val, y_val))

    return result

def stitch_data(data1, data2, cutoff: int):
    graph_range = (1,64) # TODO allow configurable?
    xs = []
    ys = []
    y_errs = []

    for i in range(1, cutoff):
        xs.append(i)
        ys.append(data1[i]['mean'])
        y_errs.append(data1[i]['stddev'])
    for i in range(cutoff, 65):
        xs.append(i)
        ys.append(data2[i]['mean'])
        y_errs.append(data2[i]['stddev'])

    return (graph_range, xs, ys, y_errs, 'red', 'place-holder-name', '-')

def prep_model_data_for_graphing(model, name):
    return ((1, 64), [item[0] for item in model], [item[1] for item in model], [], 'black', name, '-')

def strip_graphing_data(x_range, data):
    val = list(data)
    x_vals = []
    y_vals = []
    y_errs = []
    if len(y_errs) == 0:
        for x, y in zip(data[1], data[2]):
            if x < x_range[0] or x > x_range[1]:
                continue

            x_vals.append(x)
            y_vals.append(y)
    else:
        for x, y, y_err in zip(data[1], data[2]):
            if x < x_range[0] or x > x_range[1]:
                continue

            x_vals.append(x)
            y_vals.append(y)
            y_errs.append(y_err)
        val[3] = y_errs

    val[0] = x_range
    val[1] = x_vals
    val[2] = y_vals
    return tuple(val)

fast_mulmont_cutoff = 49
mulmont_benches = go_arith_benchmarks['mulmont']
#eqn_mulmont_lof, mulmont_lof = fit_quadratic(zip([mulmont_benches['y_vals'])[:fast_mulmont_cutoff]
mulmont_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['mulmont']['non-unrolled'], 'mulmont', 'red')
mulmont_generic_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['mulmont']['generic'], 'mulmont-generic', 'blue')
setmod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['setmod']['non-unrolled'], 'setmod-non-unrolled', 'red')
setmod_generic_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['setmod']['generic'], 'setmod-generic', 'blue')
addmod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['addmod']['non-unrolled'], 'addmod-non-unrolled', 'red')
submod_non_unrolled_data = format_bench_data_for_graphing((1, 64), go_arith_benchmarks['submod']['non-unrolled'], 'submod-non-unrolled', 'red')

#scatterplot_ns_data("charts/mulmont_generic.png", "MULMONTMAX Generic Benchmarks", False, [mulmont_non_unrolled_data, mulmont_generic_data])
#scatterplot_ns_data('charts/setmod.png', 'SETMOD Benchmarks', False, [setmod_non_unrolled_data, setmod_generic_data])
#scatterplot_ns_data('charts/addmod.png', 'ADDMOD Benchmarks', False, [addmod_non_unrolled_data])
#scatterplot_ns_data('charts/submod.png', 'SUBMOD Benchmarks', False, [submod_non_unrolled_data])

mulmont_non_unrolled_small_limbs_data = format_bench_data_for_graphing((1, 12), go_arith_benchmarks['mulmont']['non-unrolled'], 'mulmont', 'red')
setmod_non_unrolled_small_limbs_data = format_bench_data_for_graphing((1, 12), go_arith_benchmarks['setmod']['non-unrolled'], 'setmod', 'green')

# linear cost for setmod up to cutoff
setmod_low_eqn, setmod_low_cost = fit_linear((0,49), setmod_non_unrolled_data)
setmod_high_eqn, setmod_high_cost = fit_quadratic((49, 64), setmod_generic_data)
setmod_model = stitch_model(setmod_low_cost, setmod_high_cost, fast_mulmont_cutoff)

mulmont_evmmax = stitch_data(go_arith_benchmarks['mulmont']['non-unrolled'], go_arith_benchmarks['mulmont']['generic'], fast_mulmont_cutoff)
setmod_evmmax = stitch_data(go_arith_benchmarks['setmod']['non-unrolled'], go_arith_benchmarks['setmod']['generic'], fast_mulmont_cutoff)

mulmont_eqn, mulmont_cost = fit_quadratic((1, 49), mulmont_evmmax)

mulmont_evmmax_model_graphing_data = prep_model_data_for_graphing(mulmont_cost, 'mulmont')
scatterplot_ns_data("charts/mulmontmax_all.png", "MULMONTMAX model", False, [mulmont_evmmax, mulmont_evmmax_model_graphing_data])

setmod_evmmax_model_graphing_data = prep_model_data_for_graphing(setmod_model, "setmod")
scatterplot_ns_data("charts/setmodmax_all.png", "SETMODMAX model", False, [setmod_evmmax, setmod_evmmax_model_graphing_data])

mulmont_evmmax_model_low_limbs_graphing_data = strip_graphing_data((1, 12), mulmont_evmmax_model_graphing_data)
import pdb; pdb.set_trace()
scatterplot_ns_data("charts/mulmontmax_low.png", "MULMONTMAX EVMMAX Benchmarks", True, [mulmont_non_unrolled_small_limbs_data, mulmont_evmmax_model_low_limbs_graphing_data])
mulmont_evmmax_model_graphing_data = list(mulmont_evmmax_model_graphing_data)

setmod_evmmax_model_low_limbs_graphing_data = strip_graphing_data((1, 12), setmod_evmmax_model_graphing_data)
scatterplot_ns_data("charts/setmodmax_low.png", "SETMODMAX EVMMAX Benchmarks", True, [setmod_non_unrolled_small_limbs_data, setmod_evmmax_model_low_limbs_graphing_data])
