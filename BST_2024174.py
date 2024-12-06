








class TreeNode:
    def __init__(self, newItem, left = None, right = None) :
        self.item = newItem
        self.left = left
        self.right = right

class BinarySearchTree :
    def __init__(self):
        self.__root = None

    def search(self, x) -> str :
        return self.__searchItem(self.__root, x, "R")

    def __searchItem(self, tNode: TreeNode, x, path: str) -> str :
        if (tNode == None) :
            return path
        elif (x == tNode.item) :
            return path
        elif (x < tNode.item) :
            return self.__searchItem(tNode.left, x, path + "0")
        else :
            return self.__searchItem(tNode.right, x, path + "1")

    def insert(self, newItem) :
        self.__root = self.__insertItem(self.__root, newItem)

    def __insertItem(self, tNode: TreeNode, newItem) -> TreeNode :
        if (tNode == None) :
            tNode = TreeNode(newItem)
        elif (newItem < tNode.item) :
            tNode.left = self.__insertItem(tNode.left, newItem)
        elif (newItem > tNode.item) :
            tNode.right = self.__insertItem(tNode.right, newItem)
        return tNode
    def delete(self, x) :
        self.__root = self.__deleteItem(self.__root, x)

    def __deleteItem(self, tNode: TreeNode, x) -> TreeNode :
        if (tNode == None) :
            return None
        elif (x == tNode.item) :
            tNode = self.__deleteNode(tNode)
        elif (x < tNode.item) :
            tNode.left = self.__deleteItem(tNode.left, x)
        else :
            tNode.right = self.__deleteItem(tNode.right, x)
        return tNode

    def __deleteNode(self, tNode: TreeNode) -> TreeNode :
        if ((tNode.left == None) and (tNode.right == None)) :
            return None
        elif (tNode.left == None) :
            return tNode.right
        elif (tNode.right == None):
            return tNode.left
        else :
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.right)
            tNode.item = rtnItem
            tNode.right = rtnNode
            return tNode

    def __deleteMinItem(self, tNode: TreeNode) -> tuple :
        if (tNode.left == None) :
            return (tNode.item, tNode.right)
        else :
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.left)
            tNode.left = rtnNode
            return (rtnItem, tNode)

    def isEmpty(self) -> bool :
        return self.__root == None

    def clear(self) :
        self.__root = None


def process_input_output(readFile, writeFile) :
    lines = readFile.readlines()
    line = 0
    t = int(lines[line].strip())
    line += 1

    for j in range(t) :
        bst = BinarySearchTree()


        i = int(lines[line].strip())
        line += 1

        keys = lines[line].strip().split()

        for j in range(i) :
            key = int(keys[j])
            bst.insert(key)

        line += 1


        s1 = int(lines[line].strip())
        line += 1

        keys = lines[line].strip().split()

        for j in range(s1) :
            key = int(keys[j])
            path = bst.search(key)
            writeFile.write(path + "\n")

        line += 1


        d = int(lines[line].strip())
        line += 1

        keys = lines[line].strip().split()

        for j in range(d):
            key = int(keys[j])
            bst.delete(key)

        line += 1


        s2 = int(lines[line].strip())
        line += 1

        line_content = lines[line].strip()
        keys = line_content.split()

        for index in range(s2):
            key = int(keys[index])
            path = bst.search(key)
            writeFile.write(path + "\n")

        line += 1


readFile = open('bst_input.txt', 'r')
writeFile = open('bst_output.txt', 'w')
process_input_output(readFile, writeFile)
readFile.close()
writeFile.close()
