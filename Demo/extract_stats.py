import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='file path')
    args = parser.parse_args()
    return args
    

def print_stats(file_path, ignore_first=50):
    max_allocated = -1
    elapsed_time = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if 'max allocated:' in line:
                max_memory = float(line.split('|')[2].split(':')[1].strip())
                max_allocated = max(max_allocated, max_memory)
            
            
            if 'elapsed time per iteration' in line:
                if ignore_first > 0:
                    ignore_first -= 1
                    continue
                    
                target_part = line.split('|')[2]
                assert 'elapsed time per iteration' in target_part, 'incorrect part extracted for elapsed time'
                time = float(target_part.split(':')[1].strip())
                elapsed_time.append(time)
                
    
    print(f'max allocated memory/MB: {round(max_allocated, 2)}')
    print(f'average elapsed time/s: {round(sum(elapsed_time)/len(elapsed_time), 2)}')
    print(f'effective count for elapsed time: {len(elapsed_time)}')
        
    
def main():
    args = parse_args()
    print_stats(args.file)
    

if __name__ == '__main__':
    main()
    
    