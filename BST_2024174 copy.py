











class TreeNode:
    def __init__(self, newItem, left, right):
        self.item = newItem  # 노드의 값 (Key)
        self.left = left  # 왼쪽 자식 노드에 대한 참조
        self.right = right  # 오른쪽 자식 노드에 대한 참조

class BinarySearchTree:
    def __init__(self):
        self.__root = None  # 초기 루트 노드를 None으로 설정

    def search(self, x) -> TreeNode:
        # 주어진 값 x를 가진 노드를 찾고, 탐색 경로를 반환하는 메소드
        return self.__searchItem(self.__root, x, "R")

    def __searchItem(self, tNode: TreeNode, x, path: str) -> str:
        # 재귀적으로 노드를 탐색하는 내부 메소드
        if tNode is None:
            return path  # 탐색 실패 시 현재 경로를 반환
        elif x == tNode.item:
            return path  # 값을 찾았을 때 경로 반환
        elif x < tNode.item:
            return self.__searchItem(tNode.left, x, path + "0")  # 값이 작을 경우 왼쪽 자식 노드로 이동
        else:
            return self.__searchItem(tNode.right, x, path + "1")  # 값이 클 경우 오른쪽 자식 노드로 이동

    def insert(self, newItem):
        # 새로운 값을 삽입하는 메소드
        self.__root = self.__insertItem(self.__root, newItem)

    def __insertItem(self, tNode: TreeNode, newItem) -> TreeNode:
        # 재귀적으로 삽입 위치를 찾아 값을 삽입하는 내부 메소드
        if tNode is None:
            tNode = TreeNode(newItem)  # 빈 위치에 새로운 노드를 생성
        elif newItem == tNode.item:
            return None  # 중복된 값은 삽입하지 않음
        elif newItem < tNode.item:
            tNode.left = self.__insertItem(tNode.left, newItem)  # 왼쪽 서브트리에 삽입
        else:
            tNode.right = self.__insertItem(tNode.right, newItem)  # 오른쪽 서브트리에 삽입
        return tNode  # 수정된 트리의 루트를 반환

    def delete(self, x):
        # 주어진 값 x를 가진 노드를 삭제하는 메소드
        self.__root = self.__deleteItem(self.__root, x)

    def __deleteItem(self, tNode: TreeNode, x) -> TreeNode:
        # 재귀적으로 노드를 찾아 삭제하는 내부 메소드
        if tNode is None:
            return None  # 삭제할 노드를 찾지 못한 경우
        elif x == tNode.item:
            tNode = self.__deleteNode(tNode)  # 해당 노드를 삭제
        elif x < tNode.item:
            tNode.left = self.__deleteItem(tNode.left, x)  # 왼쪽 서브트리에서 삭제
        else:
            tNode.right = self.__deleteItem(tNode.right, x)  # 오른쪽 서브트리에서 삭제
        return tNode  # 수정된 트리의 루트를 반환

    def __deleteNode(self, tNode: TreeNode) -> TreeNode:
        # 실제 노드를 삭제하는 내부 메소드
        if tNode.left is None and tNode.right is None:
            return None  # 자식 노드가 없는 경우, 단순히 노드를 제거
        elif tNode.left is None:
            return tNode.right  # 왼쪽 자식이 없고 오른쪽 자식만 있는 경우, 오른쪽 자식을 반환
        elif tNode.right is None:
            return tNode.left  # 오른쪽 자식이 없고 왼쪽 자식만 있는 경우, 왼쪽 자식을 반환
        else:
            # 양쪽 자식이 모두 있는 경우, 오른쪽 서브트리에서 가장 작은 값을 찾아 대체
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.right)
            tNode.item = rtnItem
            tNode.right = rtnNode
            return tNode

    def __deleteMinItem(self, tNode: TreeNode) -> tuple:
        # 오른쪽 서브트리에서 가장 작은 값을 찾아 반환하는 내부 메소드
        if tNode.left is None:
            return (tNode.item, tNode.right)  # 최소값을 가진 노드와 그 오른쪽 자식을 반환
        else:
            (rtnItem, rtnNode) = self.__deleteMinItem(tNode.left)  # 왼쪽 서브트리에서 최소값을 찾음
            tNode.left = rtnNode
            return (rtnItem, tNode)  # 최소값과 수정된 노드를 반환

    def isEmpty(self) -> bool:
        # 트리가 비어 있는지 여부를 확인
        return self.__root is None

    def clear(self):
        # 트리를 초기화
        self.__root = None

def process_input_output(readFile, writeFile):
    # 파일을 읽어서 입력을 처리하고 결과를 출력하는 함수
    lines = readFile.readlines()
    idx = 0
    t = int(lines[idx].strip())  # 테스트 케이스 개수
    idx += 1

    for _ in range(t):
        bst = BinarySearchTree()

        # Insert keys
        i = int(lines[idx].strip())  # 삽입할 키의 개수
        idx += 1
        keys_to_insert = list(map(int, lines[idx].strip().split()))
        idx += 1
        for key in keys_to_insert:
            bst.insert(key)  # 키를 트리에 삽입

        # Search keys before deletion
        s1 = int(lines[idx].strip())  # 삭제 전 검색할 키의 개수
        idx += 1
        keys_to_search_1 = list(map(int, lines[idx].strip().split()))
        idx += 1
        for key in keys_to_search_1:
            path = bst.search(key)  # 검색 경로를 반환
            writeFile.write(path + "\n")

        # Delete keys
        d = int(lines[idx].strip())  # 삭제할 키의 개수
        idx += 1
        keys_to_delete = list(map(int, lines[idx].strip().split()))
        idx += 1
        for key in keys_to_delete:
            bst.delete(key)  # 키를 트리에서 삭제

        # Search keys after deletion
        s2 = int(lines[idx].strip())  # 삭제 후 검색할 키의 개수
        idx += 1
        keys_to_search_2 = list(map(int, lines[idx].strip().split()))
        idx += 1
        for key in keys_to_search_2:
            path = bst.search(key)  # 검색 경로를 반환
            writeFile.write(path + "\n")

# 실행 부분
readFile = open('bst_input.txt', 'r')
writeFile = open('bst_output.txt', 'w')
process_input_output(readFile, writeFile)
readFile.close()
writeFile.close()
