# 3) from 패키지명 import 모듈명 as 별칭, 모듈명2, ...

from sample1 import module1, module2

if __name__ == '__main__':
    print(module1.num)
    module1.fun()
    c = module1.Person()
    print(module2.num2)
    module2.fun2()
    c2 = module2.Person2()