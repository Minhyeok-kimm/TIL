-- projection (컬럼 선택)
-- 1. 테이블 보기
SELECT * FROM TAB;
SELECT * FROM user_tables;

-- 2. 테이블 정보 보기(DBeaver에서는 desc 예약어 사용 불가. 테이블 선택 후 F4 클릭)
DESC scott.dept;

-- 3. 모든 컬럼 보기
SELECT * FROM scott.dept;
SELECT * FROM scott.emp;

-- 4. 컬럼명 명시(지정 순서대로 출력)
SELECT deptno, loc FROM scott.dept;
SELECT loc, deptno FROM scott.dept;

-- 5. 컬럼명 별칭(컬럼명 as 별칭. as는 생략 가능, 별칭에 띄어쓰기 사용시 ""로 묶어야 함)
-- SQL 문에서 "" 사용하는 경우는 별칭이 유일
SELECT deptno AS 부서번호, loc AS 위치 FROM scott.dept;
SELECT deptno AS "부서 번호", loc AS 위치 FROM scott.dept;

-- 6. 연산 가능
SELECT empno, sal, sal + 100 AS 보너스 FROM scott.emp;

-- 7. 연결 가능, ||(파이프 연산자)
-- 값?  수치: 10 ,  3.15  문자(문자열):   'A' , '홍', 'ABC', '홍길동'
SELECT ename || sal AS 이름과월급 FROM scott.emp;
SELECT deptno || ename || sal FROM scott.emp;
SELECT ename || ' 사원' FROM scott.emp;

-- 8. 중복값 제거 distinct
-- emp에 어떤 job이 있는지 확인
SELECT job FROM scott.emp;
SELECT DISTINCT job FROM scott.emp;

-- select distinct 컬럼명, 컬럼명 as 별칭, 컬럼(값) || 컬럼(값), 컬럼 + 100 from 테이블명
-- oracle에서는 from 절 필수
-- select ~ from 절은 항상 모든 레코드가 출력

-- 9. NULL 값 연산
SELECT empno, sal, comm FROM scott.emp;
SELECT empno, sal, comm, (sal*12) + comm FROM scott.emp;
SELECT empno, sal, comm, (sal*12) + nvl(comm, 0) FROM scott.emp;

-- SELECTION (행 선택)
-- 10. WHERE 절
-- 가. 연산자
SELECT empno, ename, sal FROM scott.emp WHERE sal = 800;

-- 문제: 이름이 ford인 사람 찾기
-- 식별자를 제외한 값은 대소문자 구별(MariaDB는 구별하지 않음)
SELECT empno, ename, sal FROM scott.emp WHERE ename = 'FORD';


-- 문제: 입사일이 80/12/17인 사원 찾기
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE hiredate = '80/12/17';

-- 나. 범위: between A and B: A와 B는 포함. 수치, 날짜 데이터를 지정 가능함(날짜 데이터는 내부적으로 수치처리가 되기 때문)
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE sal BETWEEN 800 AND 2000;
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE hiredate BETWEEN '80/01/10' AND '80/12/31';

-- 다. 한꺼번에 여러 값 지정
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE sal IN (800, 1500, 2000);
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE ename IN ('SMITH', 'FORD');
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE hiredate IN ('80/12/17', '80/12/01');

-- 라. null 값 찾기. IS NULL 사용
SELECT empno, ename, sal, hiredate, comm FROM scott.emp WHERE comm IS NULL;
SELECT empno, ename, sal, hiredate, comm FROM scott.emp WHERE comm IS NOT NULL;

-- 마. 비슷한 값 찾기 like + % 또는 _
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE ename LIKE 'A%';
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE ename LIKE '%T%';
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE ename LIKE '%S';
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE ename LIKE '_L%';

-- 문제: 이름이 5글자이고 마지막은 N으로 끝나는 사원 조회
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE ename LIKE '____N';

-- 바. 논리 연산자: and(그리고, 모두 만족할 시 반환), or(또는, 하나만 만족해도 반환), not(부정, 반대로 반환)
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE job = 'SALESMAN' AND sal >= 1500;
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE job = 'SALESMAN' or sal >= 1500;
SELECT empno, ename, sal FROM scott.emp WHERE NOT ename = 'FORD';
SELECT empno, ename, sal, hiredate FROM scott.emp WHERE NOT ename IN ('SMITH', 'FORD');

-- 11. 정렬
-- 1) order by 컬럼명
SELECT empno, ename, sal, hiredate FROM scott.emp ORDER BY sal; -- 명시적으로 오름차순 정렬할 경우 ASC 입력
SELECT empno, ename, sal, hiredate FROM scott.emp ORDER BY sal desc;

-- 2) order by 별칭
SELECT empno, ename, sal AS salary, hiredate FROM scott.emp ORDER BY salary;

-- 3) order by 컬럼순서
SELECT empno, ename, sal AS salary, hiredate FROM scott.emp ORDER BY 3;

-- 4) 다중 정렬: order by 컬럼명1, 컬럼명2
SELECT empno, ename, sal AS salary, hiredate FROM scott.emp ORDER BY salary, hiredate DESC;
SELECT empno, ename, sal AS salary, hiredate FROM scott.emp ORDER BY 3, 4 DESC;
