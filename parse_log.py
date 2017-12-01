import re

regex_moves = re.compile('avg no of moves: ([\d.]+)')
regex_mean_q = re.compile('mean q value: ([\d.]+)')

if __name__ == '__main__':
    with open('data', 'r') as f:
        for line in f:
            if 'Perf' in line:
                avg_moves = regex_moves.search(line).group(1)
                mean_q = regex_mean_q.search(line).group(1)
                print('{},{}'.format(avg_moves, mean_q))
