








def replacementSelection(readFile, writeFile) :
    readNum = int(readFile.readline().strip())
    results = []

    for i in range(readNum) :
        n = int(readFile.readline().strip())
        numbers = []
        line = readFile.readline().strip()
        parts = line.split()
        for i in range(n) :
            numbers.append(int(parts[i]))

        buffer = []
        runs = []
        currentRun = []
        smallest = None
        
        if len(numbers) <= 5 :
            buffer.extend(numbers)
            buffer.sort()
            currentRun.extend(buffer)
            runs.append(currentRun)
        else :
            for i in range(len(numbers)) :
                num = numbers[i]
                if len(buffer) < 5 :
                    buffer.append(num)
                    buffer.sort()
                else:
                    if smallest is None :
                        smallest = buffer.pop(0)
                    else:
                        nextNum = None
                        for value in buffer :
                            if not currentRun or value >= currentRun[-1] :
                                nextNum = value
                                break

                        if nextNum is None :
                            runs.append(list(currentRun))
                            currentRun = []
                            smallest = buffer.pop(0)
                        else :
                            buffer.remove(nextNum)
                            smallest = nextNum

                    if not currentRun or smallest >= currentRun[-1] :
                        currentRun.append(smallest)

                    buffer.append(num)
                    buffer.sort()

            for i in range(len(buffer)) :
                nextNum = None
                for value in buffer:
                    if not currentRun or value >= currentRun[-1] :
                        nextNum = value
                        break

                if nextNum is None :
                    runs.append(list(currentRun))
                    currentRun = []
                    smallest = buffer.pop(0)
                else :
                    buffer.remove(nextNum)
                    smallest = nextNum

                if not currentRun or smallest >= currentRun[-1] :
                    currentRun.append(smallest)

        if currentRun:
            runs.append(currentRun)

        doubleRuns = []
        for run in runs:
            if run not in doubleRuns:
                doubleRuns.append(run)

        results.append(str(len(doubleRuns)))
        for run in doubleRuns:
            results.append(" ".join(map(str, run)))

    writeFile.write("\n".join(results))


readFile = open('replacement_input.txt', 'r')
writeFile = open('replacement_output.txt', 'w')
replacementSelection(readFile, writeFile)
readFile.close()
writeFile.close()
