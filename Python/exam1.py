# module1.py 접근하여 연동하는 파일

# 경로지정 방법
# 1) import 패키지명.모듈명

import sample1.module1



# print(__name__)
# 관례적으로 항상 시작해야하는 모듈에 지정한다.
if __name__ == '__main__':
    print(sample1.module1.num)
    sample1.module1.fun()
    c = sample1.module1.Person()