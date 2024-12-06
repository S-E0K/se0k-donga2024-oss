




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
            return path  # 노드 없을 때
        elif (x == tNode.item) :
            return path  # 키 찾으면
        elif (x < tNode.item) :
            return self.__searchItem(tNode.left, x, path + "0")  # 값이 작으면 왼쪽
        else :
            return self.__searchItem(tNode.right, x, path + "1")  # 값이 크면 오른쪽

    def insert(self, newItem) :
        self.__root = self.__insertItem(self.__root, newItem)

    def __insertItem(self, tNode: TreeNode, newItem) -> TreeNode :
        if (tNode == None) :
            tNode = TreeNode(newItem)  # 노드가 없으면
        elif (newItem < tNode.item) :
            tNode.left = self.__insertItem(tNode.left, newItem)  # 왼쪽 서브트리에 삽입
        elif (newItem > tNode.item) :
            tNode.right = self.__insertItem(tNode.right, newItem)  # 오른쪽 서브트리에 삽입
        return tNode  # 수정된 트리 노드 반환

    def delete(self, x) :
        self.__root = self.__deleteItem(self.__root, x)

    def __deleteItem(self, tNode: TreeNode, x) -> TreeNode :
        if (tNode == None) :
            return None  # 없을 때
        elif (x == tNode.item) :
            tNode = self.__deleteNode(tNode)  # 있을 때
        elif (x < tNode.item) :
            tNode.left = self.__deleteItem(tNode.left, x)  # 왼쪽 서브트리에서 삭제
        else :
            tNode.right = self.__deleteItem(tNode.right, x)  # 오른쪽 서브트리에서 삭제
        return tNode  # 수정된 트리 노드 반환

    def __deleteNode(self, tNode: TreeNode) -> TreeNode :
        if ((tNode.left == None) and (tNode.right == None)) :
            return None  # 자식이 없음
        elif (tNode.left == None) :
            return tNode.right  # 왼쪽 자식이 없음
        elif (tNode.right == None):
            return tNode.left  # 오른쪽 자식이 없음
        else :
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.right) # 둘 다 있음
            tNode.item = rtnItem
            tNode.right = rtnNode
            return tNode

    def __deleteMinItem(self, tNode: TreeNode) -> tuple :
        if (tNode.left == None) :
            return (tNode.item, tNode.right)  # 노드와 오른쪽 자식을 반환
        else :
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.left)
            tNode.left = rtnNode
            return (rtnItem, tNode)

    def isEmpty(self) -> bool :
        return self.__root == None

    def clear(self) :
        self.__root = None


def process_input_output(readFile, writeFile) : # 서치
    lines = readFile.readlines()
    line = 0
    t = int(lines[line].strip())  # 테스트 케이스 개수
    line += 1

    for j in range(t) :
        bst = BinarySearchTree()


        i = int(lines[line].strip())  # 삽입할 키의 개수
        line += 1

        keys = lines[line].strip().split()  # 문자열을 공백을 기준으로 나누어 리스트로 변환

        for j in range(i) :
            key = int(keys[j])  # 문자열 요소를 정수로 변환
            bst.insert(key)  # 정수로 변환된 키를 트리에 삽입

        line += 1


        s1 = int(lines[line].strip())  # 삭제 전 검색할 키의 개수
        line += 1

        keys = lines[line].strip().split()  # 현재 줄의 내용을 읽고 공백을 기준으로 나누어 리스트로 변환

        for j in range(s1) :
            key = int(keys[j])  # 문자열 요소를 정수로 변환
            path = bst.search(key)  # BST에서 해당 키를 검색
            writeFile.write(path + "\n")  # 검색 경로를 출력 파일에 기록

        line += 1


        d = int(lines[line].strip())  # 삭제할 키의 개수
        line += 1

        keys = lines[line].strip().split()  # 현재 줄의 내용을 읽고 공백을 기준으로 나누어 리스트로 변환

        for j in range(d):
            key = int(keys[j])  # 문자열 요소를 정수로 변환
            bst.delete(key)  # 정수로 변환된 키를 트리에서 삭제

        line += 1


        s2 = int(lines[line].strip())  # 삭제 후 검색할 키의 개수
        line += 1

        line_content = lines[line].strip()
        keys = line_content.split()  # 현재 줄의 내용을 읽고 공백을 기준으로 나누어 리스트로 변환

        for index in range(s2):
            key = int(keys[index])  # 문자열 요소를 정수로 변환
            path = bst.search(key)  # BST에서 해당 키를 검색
            writeFile.write(path + "\n")  # 검색 경로를 출력 파일에 기록

        line += 1


# 실행 부분
readFile = open('bst_input.txt', 'r')
writeFile = open('bst_output.txt', 'w')
process_input_output(readFile, writeFile)
readFile.close()
writeFile.close()
