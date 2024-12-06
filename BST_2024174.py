











class TreeNode:
    def __init__(self, newItem, left, right) :
        self.item = newItem
        self.left = left
        self.right = right

class BinarySearchTree :
    def __init__(self):
        self.__root = None

    def search(self, x) -> TreeNode :
        return self.__searchItem(self.__root, x)

    def __searchItem(self, tNode: TreeNode, x) -> TreeNode :
        if (tNode == None) :
            return None  # 노드 없을 때
        elif (x == tNode.item) :
            return tNode  # 키 찾으면
        elif (x < tNode.item) :
            return self.__searchItem(tNode.left, x)  # 값이 작으면 왼쪽
        else :
            return self.__searchItem(tNode.right, x)  # 값이 크면 오른쪽

    def insert(self, newItem) :
        self.__root = self.__insertItem(self.__root, newItem)

    def __insertItem(self, tNode: TreeNode, newItem) -> TreeNode :
        if (tNode == None) :
            tNode = TreeNode(newItem)  # 노드가 없으면
        elif (newItem == tNode.item) :
            return None  # 이미 노드가 있으면
        elif (newItem < tNode.item) :
            tNode.left = self.__insertItem(tNode.left, newItem)  # 왼쪽 서브트리에 삽입
        else :
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
            # 둘 다 있음
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.right)
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
        return self.__root == self.NIL

    def clear(self) :
        self.__root = self.NIL

def process_input_output(readFile, writeFile) : # 서치
    lines = readFile.readlines()
    line = 0
    t = int(lines[line].strip())  # 테스트 케이스 개수
    line += 1

    for j in range(t) :
        bst = BinarySearchTree()

        i = int(lines[line].strip())  # 삽입할 키의 개수
        line += 1

        # 현재 줄에서 삽입할 키들을 읽어 리스트로 변환하는 과정을 단계별로 수행
        line_content = lines[line].strip()  # 현재 줄의 내용을 읽고 앞뒤 공백 제거
        parts = line_content.split()  # 문자열을 공백을 기준으로 나누어 리스트로 변환
        keys_to_insert = []  # 삽입할 키를 저장할 빈 리스트 생성

        # parts 리스트에 있는 각 문자열 요소를 정수로 변환하여 keys_to_insert 리스트에 추가
        for part in parts:
            keys_to_insert.append(int(part))

        line += 1

        # 삽입할 키들을 트리에 삽입
        for key in keys_to_insert:
            bst.insert(key)  # 키를 트리에 삽입


        # Search keys before deletion
        s1 = int(lines[line].strip())  # 삭제 전 검색할 키의 개수
        line += 1
        keys_to_search_1 = list(map(int, lines[line].strip().split()))
        line += 1
        for key in keys_to_search_1:
            path = bst.search(key)  # 검색 경로를 반환
            writeFile.write(path + "\n")

        # Delete keys
        d = int(lines[line].strip())  # 삭제할 키의 개수
        line += 1
        keys_to_delete = list(map(int, lines[line].strip().split()))
        line += 1
        for key in keys_to_delete:
            bst.delete(key)  # 키를 트리에서 삭제

        # Search keys after deletion
        s2 = int(lines[line].strip())  # 삭제 후 검색할 키의 개수
        line += 1
        keys_to_search_2 = list(map(int, lines[line].strip().split()))
        line += 1
        for key in keys_to_search_2:
            path = bst.search(key)  # 검색 경로를 반환
            writeFile.write(path + "\n")

# 실행 부분
readFile = open('bst_input.txt', 'r')
writeFile = open('bst_output.txt', 'w')
process_input_output(readFile, writeFile)
readFile.close()
writeFile.close()
