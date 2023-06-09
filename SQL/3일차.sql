-- 2) 그룹함수
-- *은 COUNT만 사용 가능
SELECT MAX(SAL), MIN(SAL), SUM(SAL), AVG(SAL), COUNT(*), COUNT(SAL) FROM SCOTT.EMP;

-- COUNT(컬럼)은 NULL 제외
SELECT COUNT(COMM), COUNT(*) FROM SCOTT.EMP;

-- 그룹으로 묶지 않은 일반 컬럼은 그룹합수와 같이 사용할 수 없음
-- 해결하기 위해서는 일반 컬럼을 그룹으로 만들어주면 된다.(GROUP BY 사용)
SELECT DEPTNO, MAX(SAL) FROM SCOTT.EMP; -- 에러 발생

-- GROUP BY를 이용하여 부서별 최대 SAL 값 조회
SELECT DEPTNO, MAX(SAL) FROM SCOTT.EMP GROUP BY DEPTNO;

-- 부서, 직업 별 최대 SAL 값 조회
SELECT DEPTNO, JOB, MAX(SAL) FROM EMP GROUP BY DEPTNO, JOB ORDER BY DEPTNO;

-- 필터링(HAVING 절)
SELECT DEPTNO, MAX(SAL), COUNT(*) FROM SCOTT.EMP WHERE SAL < 5000
GROUP BY DEPTNO HAVING COUNT(*) > 2 ORDER BY MAX(SAL) DESC;

-- WHERE절에는 GROUP 함수 사용 불가
select deptno, max(sal), count(*) from emp where max(sal) < 5000;

-- ORDER BY vs GROUP BY
-- ORDER BY 표현식(컬럼명|별칭|순서)
-- GROUP BY 표현식(컬럼명) MariaDB는 별칭 됨
select deptno as no, max(sal), count(*)
from emp
where sal < 5000
group by deptno
having count(*) > 2
--order by deptno desc;
order by no;

-- 13. JOIN
-- ANSI JOIN만 교육 진행 예정
-- 제약조건 확인
SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME = 'EMP';
SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME = 'DEPT';

-- 1)  INNER JOIN
-- (1) natural join : 공통 컬럼 기반
-- DEPTNO 40은 출력되지 않음(공통으로 있지 않기 때문) => INNER JOIN에 해당(일치하는 값만 출력)
SELECT * FROM SCOTT.EMP NATURAL JOIN SCOTT.DEPT; -- 결과는 논리적인 가상의 테이블이 출력된다

-- 필터링
SELECT * FROM SCOTT.EMP NATURAL JOIN SCOTT.DEPT WHERE DEPTNO = 30; -- FROM 절에 JOIN 조건, WHERE 절에 검색조건
SELECT DEPTNO, MAX(SAL) FROM SCOTT.EMP NATURAL JOIN SCOTT.DEPT GROUP BY DEPTNO ORDER BY DEPTNO;

-- 테이블 별칭 지정 (AS 사용 불가)
SELECT * FROM SCOTT.EMP E NATURAL JOIN SCOTT.DEPT D;

-- 컬럼명 지정시 테이블 정보를 추가
-- JOIN 대상인 컬럼은 테이블 정보를 추가하면 안된다.
-- 테이블 별칭 지정 시 반드시 별칭으로만 사용
SELECT SCOTT.EMP.EMPNO, SCOTT.EMP.ENAME, SCOTT.EMP.SAL, SCOTT.DEPT.DNAME, SCOTT.DEPT.LOC, DEPTNO FROM SCOTT.EMP NATURAL JOIN SCOTT.DEPT;
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E NATURAL JOIN SCOTT.DEPT D;

-- (2) JOIN~ USING (공통컬럼) => INNER JOIN
SELECT EMPNO, ENAME, SAL, DNAME, LOC, DEPTNO FROM SCOTT.EMP JOIN SCOTT.DEPT USING(DEPTNO);
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E JOIN SCOTT.DEPT D USING(DEPTNO);
SELECT SCOTT.EMP.EMPNO, SCOTT.EMP.ENAME, SCOTT.EMP.SAL, SCOTT.DEPT.DNAME, SCOTT.DEPT.LOC, DEPTNO FROM SCOTT.EMP  JOIN SCOTT.DEPT USING(DEPTNO);

-- 검색조건 추가
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E JOIN SCOTT.DEPT D USING(DEPTNO)
WHERE DEPTNO = 30;

-- (3) JOIN ~ ON 조건식  => INNER JOIN, NON EQUI JOIN 가능
-- 조건식에 들어간 컬럼을 표기할 경우 테이블명 지참이 필수
SELECT EMPNO, ENAME, SAL, DNAME, LOC, SCOTT.EMP.DEPTNO
FROM SCOTT.EMP JOIN SCOTT.DEPT ON SCOTT.EMP.DEPTNO = SCOTT.DEPT.DEPTNO;

SELECT EMPNO, ENAME, SAL, DNAME, LOC, E.DEPTNO
FROM SCOTT.EMP E JOIN SCOTT.DEPT D ON E.DEPTNO = D.DEPTNO
WHERE D.DEPTNO = 10;

-- 이전까지는 모두 동등연산자로 JOIN => EQUI JOIN

-- 부등 연산자로 JOIN을 사용할 경우 반드시 ON 절 사용 => NON EQUI JOIN
SELECT * FROM SCOTT.EMP JOIN SCOTT.SALGRADE ON SAL BETWEEN LOSAL AND HISAL;
SELECT EMPNO, ENAME, SAL, GRADE FROM SCOTT.EMP JOIN SCOTT.SALGRADE ON SAL BETWEEN LOSAL AND HISAL; -- 테이블명 지정 가능, 별칭 지정 가능

SELECT SCOTT.EMP.EMPNO, SCOTT.EMP.ENAME, SCOTT.EMP.SAL, SCOTT.SALGRADE.GRADE
FROM SCOTT.EMP JOIN SCOTT.SALGRADE ON SCOTT.EMP.SAL BETWEEN SCOTT.SALGRADE.LOSAL AND SCOTT.SALGRADE.HISAL;

SELECT E.EMPNO, E.ENAME, E.SAL, S.GRADE FROM SCOTT.EMP E JOIN SCOTT.SALGRADE S ON E.SAL BETWEEN S.LOSAL AND S.HISAL;

-- (4) 3개 테이블 JOIN(EMP, DEPT, SALGRADE)
SELECT EMPNO, ENAME, DNAME, LOC, SAL, GRADE FROM SCOTT.EMP JOIN SCOTT.DEPT USING(DEPTNO)
										JOIN SCOTT.SALGRADE ON SAL BETWEEN LOSAL AND HISAL;

SELECT EMPNO, ENAME, DNAME, LOC, SAL, GRADE FROM SCOTT.EMP JOIN SCOTT.DEPT ON SCOTT.EMP.DEPTNO = SCOTT.DEPT.DEPTNO
										JOIN SCOTT.SALGRADE ON SAL BETWEEN LOSAL AND HISAL;

-- (5) SELF JOIN => INNER JOIN
-- 사원의 관리자명 조회
SELECT E.ENAME AS 사원, M.ENAME AS 관리자 FROM SCOTT.EMP E JOIN SCOTT.EMP M ON E.MGR = M.EMPNO;

-- 지금까지가 Inner 조인. 즉 일치하는 레코드만 출력된다. 일치하지 않는 레코드는 누락된다.

-- 2) OUTER JOIN
INSERT INTO scott.emp( empno, ename, sal, deptno) VALUES ( 9999, '홍길동', 500, null );
commit;

-- EMP 테이블에서는 9999 사원 누락, DEPT에서는 40 누락
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E JOIN SCOTT.DEPT D USING(DEPTNO);

-- (1) LEFT OUTER JOIN
-- EMP의 모든 데이터 출력
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E LEFT OUTER JOIN SCOTT.DEPT D USING(DEPTNO);

-- (2) RIGHT OUTER JOIN
-- DEPT의 모든 데이터 출력
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E RIGHT OUTER JOIN SCOTT.DEPT D USING(DEPTNO);

-- (3) FULL OUTER JOIN(MariaDB 지원 안 함)
-- 모든 데이터 출력
SELECT E.EMPNO, E.ENAME, E.SAL, D.DNAME, D.LOC, DEPTNO FROM SCOTT.EMP E FULL OUTER JOIN SCOTT.DEPT D USING(DEPTNO);

-- (4) 

