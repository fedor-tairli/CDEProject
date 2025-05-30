import sys
if __name__ == '__main__':
    # assert 2 arguments are given
    assert len(sys.argv) == 3, '''
    Usage:
    python ConvertListOfBranches.py <input_file> <output_file>
    '''
    with open(sys.argv[1],'r') as input, open(sys.argv[2],'w') as output:
        for line in input:
            # split line by / and take the last element
            line = line.split('/')[-1]
            output.write(line)