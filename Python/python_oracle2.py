import cx_Oracle

def db_connect():
    user = "SCOTT"
    pw = "TIGER"
    dsn = "localhost:1521/orcl" # ip번호:host번호/SID

    con = cx_Oracle.connect(user, pw, dsn, encoding="UTF-8")
    print("Database version:", con.version)
    return con

def init(con):
    while True:
        print("*"*40)
        print("1. 전체 목록 검색")
        print("2. 특정부서 검색")
        print("3. 부서 저장")
        print("4. 부서 삭제")
        print("0. 종료")
        
        # 키보드 입력
        n = int(input("메뉴 입력"))
        if n==1:
            dept_all_list(con)
        elif n==2:
            dept_by_deptno(con)
        elif n==3:
            dept_add(con)
        elif n==4:
            dept_delete(con)
        else:
            print("프로그램을 종료하였습니다")
            exit()

# 전체 목록 검색
def dept_all_list(con):
    with con.cursor() as cur:
        cur.execute("SELECT * FROM dept ORDER BY deptno")
        res = cur.fetchall()  # list로 반환
        for row in res:
            print(row)
    
# 특정 부서 검색
def dept_by_deptno(con):
    n = int(input("부서번호를 입력하시오"))
    with con.cursor() as cur:   
        cur.execute("SELECT * FROM dept WHERE deptno =:x", x=n) # 바인딩 변수(Oracle에서만 사용 가능)
        res = cur.fetchone()
        print(res)

# 저장
def dept_add(con):
    deptno = int(input("저장할 부서번호를 입력하시오"))
    dname = input("부서명을 입력하시오")
    loc = input("주소를 입력하시오")
    with con.cursor() as cur:
        cur.execute( "INSERT INTO dept (deptno, dname, loc) VALUES " \
                    " (:deptno, :dname, :loc)", deptno=deptno, dname=dname, loc=loc) # 바인딩 변수
        print("저장된 레코드갯수:", cur.rowcount) # 적용된 rowcount 불러온다
        con.commit()
        
# 삭제
def dept_delete(con):
    deptno = int(input("삭제할 부서번호를 입력하시오"))
    with con.cursor() as cur:
        cur.execute( "DELETE FROM dept WHERE deptno = :z",  z=deptno )
        print("삭제된 레코드갯수:", cur.rowcount)
        con.commit() 


if __name__ == '__main__':
    con = db_connect()
    init(con)