# 4) from 패키지명.모듈명 import 요소명, 요소명2, ..

from sample1.module1 import num, fun, Person
from sample1.module2 import num2, fun2, Person2
if __name__ == '__main__':
    print(num)
    fun()
    c = Person()

    print(num2)
    fun2()
    c2 = Person2()