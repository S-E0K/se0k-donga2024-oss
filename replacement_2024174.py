

def replacement_selection(input_file, output_file):
    f = open(input_file, 'r')
    test_num = int(f.readline().strip())
    results = []

    for i in range(test_num):
        n = int(f.readline().strip())
        numbers = list(map(int, f.readline().strip().split()))

        buffer = []
        runs = []
        current_run = []
        smallest_in_buffer = None

        for num in numbers:
            if len(buffer) < 5:
                buffer.append(num)
                buffer.sort()
            else:
                if smallest_in_buffer is None:
                    smallest_in_buffer = buffer.pop(0)
                else:
                    next_in_buffer = None
                    for i in buffer:
                        if i >= current_run[-1]:
                            next_in_buffer = i
                            break

                    if next_in_buffer is not None:
                        buffer.remove(next_in_buffer)
                        smallest_in_buffer = next_in_buffer
                    else:
                        runs.append(current_run)
                        current_run = []
                        smallest_in_buffer = buffer.pop(0)

                if not current_run or smallest_in_buffer >= current_run[-1]:
                    current_run.append(smallest_in_buffer)

                buffer.append(num)
                buffer.sort()

        while buffer:
            if smallest_in_buffer is None:
                smallest_in_buffer = buffer.pop(0)
            else:
                next_in_buffer = None
                for i in buffer:
                    if i >= current_run[-1]:
                        next_in_buffer = i
                        break

                if next_in_buffer is not None:
                    buffer.remove(next_in_buffer)
                    smallest_in_buffer = next_in_buffer
                else:
                    runs.append(current_run)
                    current_run = []
                    smallest_in_buffer = buffer.pop(0)

            if not current_run or smallest_in_buffer >= current_run[-1]:
                current_run.append(smallest_in_buffer)

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
