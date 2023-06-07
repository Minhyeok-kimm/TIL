-- 14. 서브쿼리(MariaDB와 동일)
-- 1) 단일행 서브쿼리
-- (1) where 절 서브쿼리
select empno, ename, job, sal from scott.emp where sal > ( select sal from scott.emp where ename ='WARD');

-- (2) 그룹함수 이용
select empno, ename, job, sal from SCOTT.emp where sal > ( select avg(sal) from SCOTT.emp );
SELECT AVG(SAL) FROM SCOTT.EMP;

-- 2) 복수행 서브쿼리
-- 업무(JOB)별 최소급여를 받는 사원 조회
select empno, ename, job, sal from SCOTT.emp where sal IN ( select MIN(sal) from emp Group by job );
SELECT MIN(SAL) FROM SCOTT.EMP GROUP BY JOB;

-- 업무가 MANAGER인 사원의 최소급여보다 적은 급여를 받는 사원 조회               
select empno, ename, job, sal from SCOTT.emp where sal < ALL ( select sal from SCOTT.emp where job = 'MANAGER');
select empno, ename, job, sal from SCOTT.emp where sal < ( select MIN(sal) from SCOTT.emp where job = 'MANAGER');
                  
-- 업무가 MANAGER인 사원의 최대급여보다 많은 급여를 받는 사원 조회               
select empno, ename, job, sal from SCOTT.emp where sal > ALL ( select sal from SCOTT.emp where job = 'MANAGER');
select empno, ename, job, sal from SCOTT.emp where sal > ( select MAX(sal) from SCOTT.emp where job = 'MANAGER');

-- 업무가 MANAGER인 사원의 최대급여보다 적은 급여를 받는 사원 조회               
select empno, ename, job, sal from SCOTT.emp where sal < ANY ( select sal from SCOTT.emp where job = 'MANAGER');
select empno, ename, job, sal from SCOTT.emp where sal < ( select MAX(sal) from SCOTT.emp where job = 'MANAGER');
                  
-- 업무가 MANAGER인 사원의 최소급여보다 많은 급여를 받는 사원 조회               
select empno, ename, job, sal from SCOTT.emp where sal > ANY ( select sal from SCOTT.emp where job = 'MANAGER');
select empno, ename, job, sal from SCOTT.emp where sal > ( select MIN(sal) from SCOTT.emp where job = 'MANAGER');
                  
-- exists
-- 서브쿼리 결과값이 존재하는 경우
select * from SCOTT.emp where EXISTS ( select empno from SCOTT.emp where comm is not null );
               
-- 서브쿼리 결과값이 존재하지 않는 경우
select * from SCOTT.emp where EXISTS ( select empno from SCOTT.emp where ename is null );

-- 15. DML
-- 1) INSERT 문
-- (1) 단일 생성
insert into scott.dept(deptno, dname, loc ) values ( 50, '개발', '서울');
insert into scott.dept(deptno, dname ) values ( 60, '개발');
COMMIT; -- 물리적인 파일에 반영
SELECT * FROM SCOTT.DEPT;
INSERT INTO SCOTT.DEPT VALUES(51, '개발', '서울');
COMMIT;

-- (2) 멀티 생성
-- CTAS는 NOT NULL을 제외한 제약조건이 복사가 안 됨
-- CREATE는 DDL문, DDL문은 Auto COMMIT
-- 레코드 포함
create table copy_dept AS select * from SCOTT.dept;
--레코드 미포함(테이블 구조만)
create table copy_dept2 AS select * from SCOTT.dept where 1=2;
-- INSERT ~ SELECT 문 이용해 여러개의 레코드를 INSERT(멀티 레코드 생성)
INSERT INTO SCOTT.COPY_DEPT2 SELECT DEPTNO, DNAME, LOC FROM SCOTT.DEPT;

SELECT * FROM SCOTT.COPY_DEPT2;

--  2) update 문
-- 가. 특정 레코드만 수정(일반적인 코드)
SELECT * FROM DEPT;
update dept set dname ='인사', loc='제주' where deptno = 60;

-- 나. 모든 레코드 수정
update dept set dname ='인사', loc='제주';
ROLLBACK;

-- UPDATE + 서브쿼리
UPDATE SCOTT.DEPT SET DNAME = (SELECT DNAME FROM SCOTT.DEPT WHERE DEPTNO=10),
LOC = (SELECT LOC FROM SCOTT.DEPT WHERE DEPTNO=20)
WHERE DEPTNO = 50;


-- 3) delete 문
-- 가. 특정 레코드만 삭제(일반적)
-- FK키가 참조하는 행은 삭제할 수 없음(단 옵션을 주면 삭제 가능)
delete from SCOTT.dept where deptno = 10; -- 참조 무결성 제약조건에 위배

-- 참조가 없는 행은 삭제 가능
DELETE FROM SCOTT.DEPT WHERE DEPTNO = 50;
SELECT * FROM SCOTT.DEPT;

 -- 나. 모든 레코드 삭제
DELETE FROM SCOTT.EMP;
SELECT * FROM SCOTT.EMP;
ROLLBACK;

-- DELETE + 서브쿼리
DELETE FROM SCOTT.DEPT WHERE DEPTNO IN (SELECT DEPTNO FROM DEPT WHERE DNAME = '개발');
COMMIT;

-- 16. DDL
-- 1) 테이블 생성
create table dept_2
( deptno number(2), -- 0~99
  dname varchar2(10), --  10byte
  loc varchar2(10));

 
SELECT * FROM user_tables;

SELECT * FROM SCOTT.DEPT_2;

-- 2) default  
create table dept_3
( deptno number(2),
  dname varchar2(10),
  loc varchar2(10) default '서울');

 
insert into dept_3(deptno,dname) values ( 1, '개발');
select * from dept_3;

create table dept_3_1
(deptno number(2),
dname varchar2(10),
loc varchar2(10) default '서울',
writeday DATE DEFAULT SYSDATE);

INSERT INTO DEPT_3_1 (DEPTNO, DNAME) VALUES (1, '개발');
SELECT * FROM DEPT_3_1

-- 3) 제약조건
-- 가. primary key ( 컬럼 레벨 )
-- 제약조건 이름 지정 방식
create table table1
( no number(2) constraint table1_no_pk PRIMARY KEY,
email varchar2(20));

-- 제약조건 이름 미지정 방식
create table table1_1
( no number(2) PRIMARY KEY,
email varchar2(20));

-- 추후에 제약조건을 삭제 및 비활성화 가능
SELECT * FROM user_constraints;

insert into table1 ( no, email) values ( 1, 'aaa1');
insert into table1 ( no, email) values ( 2, 'aaa2');
insert into table1 ( no, email) values ( null, 'aaa3'); -- 오류 NO는 PK


