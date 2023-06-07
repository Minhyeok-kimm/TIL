-- 16-3) 제약조건 이어서

-- 가. primary key ( 컬럼 레벨 )
DROP TABLE SCOTT.TABLE1;

create table table1
( no number(2) constraint table1_no_pk PRIMARY KEY,
  email varchar2(20));

--  primary key ( 테이블 레벨 )
create table table2
( no number(2),
  email varchar2(20),
  constraint table2_no_pk PRIMARY KEY(no)
  );

-- 제약조건 확인(WHERE절로 테이블 지정 가능)
SELECT * FROM USER_CONSTRAINTS;

 -- 복합컬럼 PK
create table table2_1
( no number(2),
  email varchar2(20),
  ADDRESS VARCHAR2(20),
  constraint table2_1_no_pk PRIMARY KEY(NO, EMAIL)
  );

-- 나. unique ( 컬럼 레벨 )
create table table3
( no number(2) constraint table3_no_uk UNIQUE,
  email varchar2(20));
 
INSERT INTO SCOTT.TABLE3(NO, EMAIL) VALUES (1, 'AAA');
insert into SCOTT.table3(no, email) values ( 2, 'aaa');
insert into SCOTT.table3(no, email) values ( null, 'aaa');

--  unique ( 테이블 레벨 )
create table table4
( no number(2),
  email varchar2(20),
  constraint table4_no_uk UNIQUE(no)
  );

 -- 복합컬럼 UK
create table table4_1
( no number(2),
  email varchar2(20),
  ADDRESS VARCHAR2(20),
  constraint table4_1_no_Uk UNIQUE(NO, EMAIL)
  );

-- 다. not null ( 컬럼 레벨 )
create table table7
( no number(2) constraint table7_no_nn NOT NULL,
  email varchar2(20));

-- not null (테이블 레벨 지원 안 됨)

-- 라. check ( 컬럼 레벨 )
create table table5
( no number(2) constraint table5_no_ck CHECK(no in (10,20)),
  email varchar2(20));
 
insert into table5 ( no, email) values (10, 'aa');
insert into table5 ( no, email) values (20, 'aa');
insert into table5 ( no, email) values (30, 'aa');

--  check ( 테이블 레벨 )
create table table6
( no number(2),
  email varchar2(20),
  constraint table6_no_ck CHECK(no >= 20)
  );
  
insert into table6 ( no, email) values (10, 'aa');
insert into table6 ( no, email) values (20, 'aa');
insert into table6 ( no, email) values (30, 'aa');

-- 마. foreign key
-- 부모 테이블 생성
CREATE TABLE MASTER1 (
NUM NUMBER(2) CONSTRAINT MASTER_NUM_PK PRIMARY KEY,
EMAIL VARCHAR2(10));

INSERT INTO SCOTT.MASTER1 (NUM, EMAIL) VALUES (1, 'aa1');
INSERT INTO SCOTT.MASTER1 (NUM, EMAIL) VALUES (2, 'aa2');
INSERT INTO SCOTT.MASTER1 (NUM, EMAIL) VALUES (3, 'aa3');
COMMIT;

-- 자식 테이블 생성
CREATE TABLE SLAVE1 (
NO NUMBER(2) CONSTRAINT SLAVE1_NO_PK PRIMARY KEY,
NAME VARCHAR2(10),
NUM NUMBER CONSTRAINT SLAVE1_NUM_FK REFERENCES MASTER1(NUM));

INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (10, 'xxx1', 1);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (20, 'xxx2', 2);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (30, 'xxx3', 3);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (40, 'xxx4', 4); -- 에러 발생
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (50, 'xxx5', NULL);
COMMIT;

CREATE TABLE SLAVE2 (
NO NUMBER(2) CONSTRAINT SLAVE2_NO_PK PRIMARY KEY,
NAME VARCHAR2(10),
NUM NUMBER(2),
CONSTRAINT SLAVE2_NUM_FK FOREIGN KEY(NUM) REFERENCES MASTER1(NUM));

INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (10, 'xxx1', 1);
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (20, 'xxx2', 2);
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (30, 'xxx3', 3);
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (40, 'xxx4', 4); -- 에러 발생
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (50, 'xxx5', NULL);
COMMIT;

-- MASTER1의 NUM = 1인 레코드 삭제
DELETE FROM MASTER1 WHERE NUM = 1; -- 불가능

DROP TABLE SLAVE1;
DROP TABLE SLAVE2;

-- FK 옵션 지정
CREATE TABLE SLAVE1 (
NO NUMBER(2) CONSTRAINT SLAVE1_NO_PK PRIMARY KEY,
NAME VARCHAR2(10),
NUM NUMBER CONSTRAINT SLAVE1_NUM_FK REFERENCES MASTER1(NUM) ON DELETE CASCADE);

INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (10, 'xxx1', 1);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (20, 'xxx2', 2);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (30, 'xxx3', 3);
-- INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (40, 'xxx4', 4); -- 에러 발생
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (50, 'xxx5', NULL);
COMMIT;

SELECT * FROM MASTER1;
SELECT * FROM SLAVE1;

DELETE FROM MASTER1 WHERE NUM = 1;
ROLLBACK;

CREATE TABLE SLAVE2 (
NO NUMBER(2) CONSTRAINT SLAVE2_NO_PK PRIMARY KEY,
NAME VARCHAR2(10),
NUM NUMBER(2),
CONSTRAINT SLAVE2_NUM_FK FOREIGN KEY(NUM) REFERENCES MASTER1(NUM) ON DELETE SET NULL);

-- INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (10, 'xxx1', 1);
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (20, 'xxx2', 2);
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (30, 'xxx3', 3);
--INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (40, 'xxx4', 4); -- 에러 발생
INSERT INTO SLAVE2 (NO, NAME, NUM) VALUES (50, 'xxx5', NULL);
COMMIT;

SELECT * FROM MASTER1;
SELECT * FROM SLAVE2;

DELETE FROM MASTER1 WHERE NUM = 2;


-- 부모 테이블 삭제
DROP TABLE MASTER1;

SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME = 'SLAVE2';

DROP TABLE MASTER1 CASCADE CONSTRAINTS; -- 실행 후 SLAVE2 FK 제약 조건 삭제

-- 실습
DROP TABLE SUGANG;
DROP TABLE STUDENT;
DROP TABLE SUBJECT;

-- TYPE 행에서 '필수'가 오라클에서는 6byte, MariaDB는 4byte로 인식
CREATE TABLE SUBJECT (
SUBNO NUMBER(5) CONSTRAINT SUBJECT_SUBNO_PK PRIMARY KEY,
SUBNAME VARCHAR2(20) CONSTRAINT SUBJECT_SUBNAME_NN NOT NULL,
TERM VARCHAR2(1) CONSTRAINT SUBJECT_TERM_CK CHECK (TERM IN (1, 2)),
TYPE VARCHAR2(6) CONSTRAINT SUBJECT_TYPE_CK CHECK (TYPE IN ('필수', '선택'))); -- 원래 답은 varchar2(4)

CREATE TABLE STUDENT (
STUDNO NUMBER(5) CONSTRAINT STUDENT_STUDNO_PK PRIMARY KEY,
STUNAME VARCHAR2(10));

CREATE TABLE SUGANG (
STUDNO NUMBER(5),
SUBNO NUMBER(5),
REGDATE DATE,
RESUT NUMBER(3),
CONSTRAINT SUGANG_NO_PK PRIMARY KEY (STUDNO, SUBNO),
CONSTRAINT SUGANG_STUDNO_FK FOREIGN KEY (STUDNO) REFERENCES STUDENT (STUDNO),
CONSTRAINT SUGANG_SUBNO_FK FOREIGN KEY (SUBNO) REFERENCES SUBJECT (SUBNO));

SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME IN ('SUBJECT', 'STUDENT', 'SUGANG');

insert into subject ( subno , subname, term , type ) values ( 1, 'a','1','필수');
insert into subject ( subno , subname, term , type ) values ( 2, 'a','3','필수');
insert into subject ( subno , subname, term , type ) values ( 3, 'a','1','필');

SELECT * FROM SUBJECT;

select * from nls_database_parameters where parameter = 'NLS_CHARACTERSET' or parameter = 'NLS_NCHAR_CHARACTERSET';
-- AL32UTF8은 한글을 3byte로 처리


-- DELETE(DML) vs TRUNCATE(DDL)
DELETE FROM SCOTT.COPY_DEPT;
SELECT * FROM SCOTT.COPY_DEPT;
ROLLBACK; -- ROLLBACK 가능

-- TRUNCATE문
SELECT * FROM SCOTT.COPY_DEPT;
TRUNCATE TABLE COPY_DEPT;
ROLLBACK; -- ROLLBACK 불가능



-- 테이블 변경
create table my_dept
as
select * from dept;

SELECT * FROM SCOTT.MY_DEPT;

-- 컬럼 추가
alter table SCOTT.my_dept
add ( tel varchar(20));

alter table SCOTT.my_dept
add ( EMAIL varchar(20), ADDR VARCHAR2(10));

-- 타입 변경 (MariaDB는 modify 컬럼 타입 -> 괄호 사용 불가)
-- 이 방식으로 not null로 수정 가능
alter table my_dept
modify ( tel number(10));

-- 컬럼 삭제 (MariaDB는 drop 컬럼 -> 괄호 사용 불가)
alter table my_dept
drop ( tel );

alter table my_dept
drop ( EMAIL, ADDR );

-- 제약조건 추가
alter table my_dept
add constraint my_dept_deptno_pk PRIMARY KEY(deptno);

alter table my_dept
add constraint my_dept_dNAME_UK UNIQUE (dNAME);

SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME = 'MY_DEPT';

-- NOT NULL 제약조건
ALTER TABLE SCOTT.MY_DEPT
MODIFY (LOC VARCHAR2(13) CONSTRAINT MY_DEPT_LOC_NN NOT NULL );

-- 제약조건 삭제 (기본적으로 CONSTRAINT 제약조건 이름 으로 삭제, primary key는 테이블당 하나이기 때문에 primary key로도 삭제 가능)
alter table my_dept
drop primary key;

alter table my_dept DROP CONSTRAINT MY_DEPT_DNAME_UK;

ALTER TABLE MY_DEPT DROP CONSTRAINT MY_DEPT_LOC_NN;

-- FK가 참조하고 있는 경우에 부모 테이블의 PK 제약조건을 삭제
CREATE TABLE MASTER1 (
NUM NUMBER(2) CONSTRAINT MASTER_NUM_PK PRIMARY KEY,
EMAIL VARCHAR2(10));

INSERT INTO SCOTT.MASTER1 (NUM, EMAIL) VALUES (1, 'aa1');
INSERT INTO SCOTT.MASTER1 (NUM, EMAIL) VALUES (2, 'aa2');
INSERT INTO SCOTT.MASTER1 (NUM, EMAIL) VALUES (3, 'aa3');
COMMIT;

CREATE TABLE SLAVE1 (
NO NUMBER(2) CONSTRAINT SLAVE1_NO_PK PRIMARY KEY,
NAME VARCHAR2(10),
NUM NUMBER CONSTRAINT SLAVE1_NUM_FK REFERENCES MASTER1(NUM));

INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (10, 'xxx1', 1);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (20, 'xxx2', 2);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (30, 'xxx3', 3);
INSERT INTO SLAVE1 (NO, NAME, NUM) VALUES (50, 'xxx5', NULL);
COMMIT;

SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME = 'MASTER1'; -- PK 제약조건
SELECT * FROM USER_CONSTRAINTS WHERE TABLE_NAME = 'SLAVE1'; -- FK 제약조건

ALTER TABLE SCOTT.MASTER1 DROP PRIMARY KEY; --삭제 불가
ALTER TABLE SCOTT.MASTER1 DROP PRIMARY KEY CASCADE; -- PK를 참조하는 FK 제약조건을 모두 삭제하면서 PK제약조건을 제거

-- 테이블 삭제
-- DROP TABLE  table명 
DROP TABLE  my_dept CASCADE CONSTRAINTS;

-- 추가질문
-- scott 계정에는 create view 권한이 없다.
-- 관리자 작업
grant create view 
to scott;

-- scott 작업
create view dept_view2
as
select deptno, dname
from dept;

select *
from dept_view2;

select * 
from user_views;
