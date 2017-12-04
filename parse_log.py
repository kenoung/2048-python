import re

regex_moves = re.compile('avg no of moves: ([\d.]+)')
regex_mean_q = re.compile('mean q value: ([\d.]+)')

if __name__ == '__main__':
    with open('result/2048-ddqn-sparse-32-0.95-0.0001-lp-de-True.log', 'r') as f:
        for line in f:
            if 'loss' in line:
                json_data = line.split('INFO ')[0]
                parsed_data = json.loads(json_data)