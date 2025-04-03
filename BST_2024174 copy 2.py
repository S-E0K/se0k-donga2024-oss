class TreeNode:
    def __init__(self, newItem, left=None, right=None):
        self.item = newItem
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.__root = None

    def search(self, x) -> str:
        return self.__searchItem(self.__root, x, "R")

    def __searchItem(self, tNode: TreeNode, x, path: str) -> str:
        if tNode is None:
            return path  # 노드가 없을 때 경로 반환
        elif x == tNode.item:
            return path  # 키를 찾았을 때 경로 반환
        elif x < tNode.item:
            # 변경점: `path = "0"` 대신, `path + "0"`을 사용하여 기존 경로에 이어붙입니다.
            return self.__searchItem(tNode.left, x, path + "0")
        else:
            # 변경점: `path = "1"` 대신, `path + "1"`을 사용하여 기존 경로에 이어붙입니다.
            return self.__searchItem(tNode.right, x, path + "1")

    def insert(self, newItem):
        self.__root = self.__insertItem(self.__root, newItem)

    def __insertItem(self, tNode: TreeNode, newItem) -> TreeNode:
        if tNode is None:
            return TreeNode(newItem)  # 노드가 없으면 새 노드 생성
        elif newItem < tNode.item:
            tNode.left = self.__insertItem(tNode.left, newItem)  # 왼쪽 서브트리에 삽입
        elif newItem > tNode.item:
            tNode.right = self.__insertItem(tNode.right, newItem)  # 오른쪽 서브트리에 삽입
        return tNode  # 수정된 트리 반환

    def delete(self, x):
        self.__root = self.__deleteItem(self.__root, x)

    def __deleteItem(self, tNode: TreeNode, x) -> TreeNode:
        if tNode is None:
            return None  # 삭제할 노드를 찾지 못한 경우
        elif x == tNode.item:
            tNode = self.__deleteNode(tNode)  # 노드를 삭제
        elif x < tNode.item:
            tNode.left = self.__deleteItem(tNode.left, x)  # 왼쪽 서브트리에서 삭제
        else:
            tNode.right = self.__deleteItem(tNode.right, x)  # 오른쪽 서브트리에서 삭제
        return tNode  # 수정된 트리 반환

    def __deleteNode(self, tNode: TreeNode) -> TreeNode:
        if tNode.left is None and tNode.right is None:
            return None  # 자식이 없는 경우
        elif tNode.left is None:
            return tNode.right  # 왼쪽 자식이 없는 경우
        elif tNode.right is None:
            return tNode.left  # 오른쪽 자식이 없는 경우
        else:
            # 둘 다 있는 경우
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.right)
            tNode.item = rtnItem
            tNode.right = rtnNode
            return tNode

    def __deleteMinItem(self, tNode: TreeNode) -> tuple:
        if tNode.left is None:
            return (tNode.item, tNode.right)  # 노드와 오른쪽 자식을 반환
        else:
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.left)
            tNode.left = rtnNode
            return (rtnItem, tNode)

    def isEmpty(self) -> bool:
        # 변경점: `self.NIL`을 `None`으로 수정했습니다.
        return self.__root is None

    def clear(self):
        # 변경점: `self.NIL`을 `None`으로 수정했습니다.
        self.__root = None

def process_input_output(readFile, writeFile):
    lines = readFile.readlines()
    line = 0
    t = int(lines[line].strip())  # 테스트 케이스 개수
    line += 1

    for _ in range(t):
        bst = BinarySearchTree()

        # Insert keys
        i = int(lines[line].strip())  # 삽입할 키의 개수
        line += 1
        keys = lines[line].strip().split()
        for j in range(i):
            key = int(keys[j])
            bst.insert(key)
        line += 1

        # Search keys before deletion
        s1 = int(lines[line].strip())  # 삭제 전 검색할 키의 개수
        line += 1
        keys = lines[line].strip().split()
        for j in range(s1):
            key = int(keys[j])
            path = bst.search(key)
            writeFile.write(path + "\n")
        line += 1

        # Delete keys
        d = int(lines[line].strip())  # 삭제할 키의 개수
        line += 1
        keys = lines[line].strip().split()
        for j in range(d):
            key = int(keys[j])
            bst.delete(key)
        line += 1

        # Search keys after deletion
        s2 = int(lines[line].strip())  # 삭제 후 검색할 키의 개수
        line += 1
        keys = lines[line].strip().split()
        for j in range(s2):
            key = int(keys[j])
            path = bst.search(key)
            writeFile.write(path + "\n")
        line += 1

# 실행 부분
readFile = open('bst_input.txt', 'r')
writeFile = open('bst_output.txt', 'w')
process_input_output(readFile, writeFile)
readFile.close()
writeFile.close()
