

def replacement_selection(input_file, output_file):
    f = open(input_file, 'r')
    test_num = int(f.readline().strip())
    results = []

    for i in range(test_num):
        n = int(f.readline().strip())
        
        line = f.readline()
        line = line.strip()
        parts = line.split()
        numbers = list(map(int, parts))

        buffer = []
        runs = []
        current_run = []
        smallest = None

        for num in numbers:
            if len(buffer) < 5:
                buffer.append(num)
                buffer.sort()
            else:
                if smallest is None:
                    smallest = buffer.pop(0)
                else:
                    next_in_buffer = None
                    for i in buffer:
                        if i >= current_run[-1]:
                            next_in_buffer = i
                            break

                    if next_in_buffer is not None:
                        buffer.remove(next_in_buffer)
                        smallest = next_in_buffer
                    else:
                        runs.append(current_run)
                        current_run = []
                        smallest = buffer.pop(0)

                if not current_run or smallest >= current_run[-1]:
                    current_run.append(smallest)

                buffer.append(num)
                buffer.sort()

        while buffer:
            if smallest is None:
                smallest = buffer.pop(0)
            else:
                next_in_buffer = None
                for i in buffer:
                    if i >= current_run[-1]:
                        next_in_buffer = i
                        break

                if next_in_buffer is not None:
                    buffer.remove(next_in_buffer)
                    smallest = next_in_buffer
                else:
                    runs.append(current_run)
                    current_run = []
                    smallest = buffer.pop(0)

            if not current_run or smallest >= current_run[-1]:
                current_run.append(smallest)

        if current_run:
            runs.append(current_run)

        results.append(f"{len(runs)}")
        for run in runs:
            results.append(" ".join(map(str, run)))

    f.close()

    f = open(output_file, 'w')
    f.write("\n".join(results))
    f.close()


replacement_selection('replacement_input.txt', 'replacement_output.txt')
