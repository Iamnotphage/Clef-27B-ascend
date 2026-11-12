#!/usr/bin/env python3
"""Render report plots from the archived profiling run; no inference is run."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run = args.run_dir
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True,
                         'grid.alpha': .2, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    colors = ['#245b97', '#cf6927']
    stable = read(run / 'stable-summary/summary.json')['groups']
    ax = axes[0, 0]
    for i, tp in enumerate((1, 2)):
        group = stable[f'tp{tp}-1024']
        rounds = group['round_requests_per_second']
        value = group['pooled_successful_requests_per_second']
        ax.bar(i, value, color=colors[i], width=.55,
               yerr=[[value - rounds['min']], [rounds['max'] - value]], capsize=5)
        ax.text(i, value + .22, f'{value:.3f}', ha='center', weight='bold')
    ax.set(xticks=[0, 1], xticklabels=['TP1', 'TP2'], ylabel='Requests / second',
           ylim=(0, 9.2), title='307 tokens: 3 rounds x 300 requests\nProfiler off; whiskers = round range')

    ax = axes[0, 1]
    for tp, color in zip((1, 2), colors):
        cases = read(run / f'tp{tp}-4096-matrix/matrix/matrix-summary.json')['cases']
        cases = sorted((row for row in cases if row['group'] == 'controlled_length'),
                       key=lambda row: row['expected_input_tokens'])
        ax.plot([row['expected_input_tokens'] for row in cases],
                [row['latency_ms']['mean'] for row in cases], 'o-', color=color, label=f'TP{tp}')
    ax.set(xlabel='Actual input tokens', ylabel='Mean request latency (ms)',
           title='Fixed schema, varying real state prefix\n30 requests per shape; profiler off')
    ax.legend()

    ax = axes[1, 0]
    for sequences, color in zip((1, 2, 4), ['#71777d', '#245b97', '#cf6927']):
        label = 'tp2-4096-matrix' if sequences == 1 else f'tp2-4096-batch{sequences}'
        values = []
        for concurrency in (1, 2, 4):
            name = 'fixed307-control.json' if sequences == 1 and concurrency == 1 else f'concurrency-{concurrency}.json'
            values.append(read(run / label / name)['summary']['successful_requests_per_second'])
        ax.plot([1, 2, 4], values, 'o-', color=color, label=f'max-num-seqs={sequences}')
    ax.set(xlabel='HTTP concurrency', ylabel='Requests / second', xticks=[1, 2, 4],
           title='TP2 / 4096: batching extension windows\n30 or 60 requests; profiler inactive')
    ax.legend()

    # One actual recorded long-input request, supplied by the trace audit.
    # The two device intervals overlap; their durations must not be added.
    ax = axes[1, 1]
    origin = 1794497050879679082
    backbone_end = 1794497051731743791
    wait_start, wait_end = 1794497051077682594, 1794497051732569121
    cpu_return = 1794497051073779700
    ms = lambda value: (value - origin) / 1e6
    ax.broken_barh([(0, ms(backbone_end))], (1.5, .5), facecolors=colors[0])
    ax.broken_barh([(ms(wait_start), ms(wait_end) - ms(wait_start))], (.5, .5), facecolors=colors[1], alpha=.8)
    ax.axvline(ms(cpu_return), color='#555', linestyle='--', linewidth=1)
    ax.text(ms(cpu_return) + 10, 2.2, 'CPU backbone scope returns', fontsize=9)
    ax.text(380, 1.68, 'Backbone device span', color='white')
    ax.text(400, .68, 'Head EVENT WAIT overlaps backbone', color='white', fontsize=9)
    ax.set(xlim=(-15, 900), ylim=(.2, 2.6), yticks=[], xlabel='Time since first backbone device task (ms)',
           title='One TP1 long-input trace: 654.061 ms overlap\nHead active compute averages 2.695 ms, reported separately')
    fig.suptitle('Clef-27B on Ascend A3 | run 20261112-223732', fontsize=15, weight='bold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
