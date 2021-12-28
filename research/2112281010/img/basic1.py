class parent1:
    def __init__(self):
        print("parent1 생성 중입니다.")
    
    def parent1_function(self):
        print("parent1의 메서드입니다.")

class parent2:
    def __init__(self):
        print("parent2 생성 중입니다.")

class functionSet:
    def method1(self):
        print("method1")

def testfunction():
    print("아무나 사용 가능")

class childBoy(parent1, parent2, functionSet): #이렇게 'functionSet` API 함수셋을 childBoy에 부여하는 형태의 디자인 패턴이 있다.
    def __init__(self):
        super().__init__() #첫번째 상속한 `parent1` 만 __init__() 이 호출됨
        print("child 생성 중입니다.")
        testfunction() #이렇게 설계하는 것은 좋은 방식은 아니다.
    
    def parent1_function(self):
        print("child 에서 재정의하였습니다.")

class childGirl(parent2, parent1):
    def __init__(self):
        super().__init__() #첫번째 상속한 `parent2` 만 __init__() 이 호출됨
        print("child 생성 중입니다.")
    
    def parent1_function(self):
        print("child 에서 재정의하였습니다.")

    # 클래스 여러개를 상속하였을 때, 상속 클래스의 순서도 중요하다.
    # super().__init__() 상속받은 클래스를 초기화하는 것인데, 위 경우에 `parent1`이 먼저 상속되었으므로 출력은 "parent1 생성 중입니다." 가 된다.

child1 = childBoy()
child1.method1()

child = childGirl()

# 오버로딩과 오버라이딩의 차이
# 오버로딩은 함수 이름은 같지만 인자가 다른 경우,
# 오버라이딩은 함수 이름과 인자가 모두 같은 경우,

