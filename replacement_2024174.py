def replacement_selection(input_file, output_file):
    with open(input_file, 'r') as f:
        test_cases = int(f.readline().strip())
        results = []

        for _ in range(test_cases):
            n = int(f.readline().strip())
            numbers = list(map(int, f.readline().strip().split()))

            buffer = []
            runs = []
            current_run = []
            smallest_in_buffer = -9999999

            for num in numbers:
                if len(buffer) < 5:
                    buffer.append(num)
                    buffer.sort()
                else:
                    if (smallest_in_buffer == -9999999) :
                        smallest_in_buffer = buffer.pop(0)
                    else :
                        for i in buffer :
                            if (smallest_in_buffer <= i) :
                                smallest_in_buffer = i
                                buffer.remove(i)
                                break
                    #smallest_in_buffer = buffer.pop(0)

                    if not current_run or smallest_in_buffer > current_run[-1]:
                        current_run.append(smallest_in_buffer)
                        
                    else:
                        runs.append(current_run)
                        current_run = [smallest_in_buffer]
                        smallest_in_buffer = -9999999
                    buffer.append(num)
                    buffer.sort()

            # Finalize the remaining items in buffer and current run
            while buffer:
                # smallest_in_buffer = buffer.pop(0)
                
                for i in buffer :
                    if (smallest_in_buffer <= i) :
                        smallest_in_buffer = i
                        buffer.remove(i)
                        break
                
                if not current_run or smallest_in_buffer > current_run[-1]:
                    current_run.append(smallest_in_buffer)
                else:
                    runs.append(current_run)
                    current_run = [smallest_in_buffer]
                    smallest_in_buffer = -9999999

            # Append last run if it has any elements
            if current_run:
                runs.append(current_run)

            # Save the results for the test case
            results.append(f"{len(runs)}")
            for run in runs:
                results.append(" ".join(map(str, run)))

    with open(output_file, 'w') as f:
        f.write("\n".join(results))


replacement_selection('replacement_input.txt', 'replacement_output.txt')
