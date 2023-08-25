-- 8/25
-- 데이터베이스 생성
CREATE DATABASE tableDB;
USE tableDB;

-- userTbl 생성
CREATE TABLE userTbl (
	userID CHAR(8),
    name CHAR(10) NOT NULL,
    birthYear INT NOT NULL,
	addr CHAR(3) NOT NULL,
    mobile1 SMALLINT,
    mobile2 INT,
    height SMALLINT NOT NULL,
    mDate DATE NOT NULL,
    CONSTRAINT PRIMARY KEY PK_userTbl_userID (userID)
);

-- buyTbl 생성 (먼저 생성하면 error 발생: userTbl이 없는 상태에서 생성하면 error)
CREATE TABLE buyTbl (
	num INT AUTO_INCREMENT PRIMARY KEY,
    userid CHAR(8) NOT NULL,
    prodName CHAR(6) NOT NULL,
    groupName CHAR(4),
    price INT NOT NULL,
    amount SMALLINT NOT NULL,
    CONSTRAINT FK_userTbl_buyTbl FOREIGN KEY(userid) REFERENCES userTbl(userID)
);

-- userTbl 데이터 insert
INSERT INTO userTbl VALUES('LSG', '이승기', 1987, '서울', '011', '1111111', 182, '2008-8-8');
INSERT INTO userTbl VALUES('KBS', '김범수', 1979, '경남', '011', '2222222', 173, '2012-4-4');
INSERT INTO userTbl VALUES('KKH', '김경호', 1971, '전남', '019', '3333333', 177, '2007-7-7');

-- buyTbl 데이터 insert
INSERT INTO buyTbl VALUES(NULL, 'KBS', '운동화', NULL, 30, 2);
INSERT INTO buyTbl VALUES(NULL, 'KBS', '노트북', '전자', 1000, 1);
INSERT INTO buyTbl VALUES(NULL, 'JYP', '모니터', '전자', 200, 1); -- JYP가 현재 정의되지 않아 오류

-- 필요한 모든 데이터 insert
INSERT INTO userTbl VALUES('JYP', '조용필', 1950, '경기', '011', '4444444', 166, '2009-4-4');
INSERT INTO userTbl VALUES('SSK', '성시경', 1979, '서울', NULL, NULL, 186, '2013-12-12');
INSERT INTO userTbl VALUES('LJB', '임재범', 1963, '서울', '016', '6666666', 182, '2009-9-9');
INSERT INTO userTbl VALUES('YJS', '윤종신', 1969, '경남', NULL, NULL, 170, '2005-5-5');
INSERT INTO userTbl VALUES('EJW', '은지원', 1972, '경북', '011', '8888888', 174, '2014-3-3');
INSERT INTO userTbl VALUES('JKW', '조관우', 1965, '경기', '018', '9999999', 172, '2010-10-10');
INSERT INTO userTbl VALUES('BBK', '바비킴', 1973, '서울', '010', '0000000', 176, '2013-5-5');
INSERT INTO buyTbl VALUES(NULL, 'JYP', '모니터', '전자', 200, 1);
INSERT INTO buyTbl VALUES(NULL, 'BBK', '모니터', '전자', 200, 5);
INSERT INTO buyTbl VALUES(NULL, 'KBS', '청바지', '의류', 50, 3);
INSERT INTO buyTbl VALUES(NULL, 'BBK', '메모리', '전자', 80, 10);
INSERT INTO buyTbl VALUES(NULL, 'SSK', '책', '서적', 15, 5);
INSERT INTO buyTbl VALUES(NULL, 'EJW', '책', '서적', 15, 2);
INSERT INTO buyTbl VALUES(NULL, 'EJW', '청바지', '의류', 50, 1);
INSERT INTO buyTbl VALUES(NULL, 'BBK', '운동화', NULL, 30, 2);
INSERT INTO buyTbl VALUES(NULL, 'EJW', '책', '서적', 15, 1);
INSERT INTO buyTbl VALUES(NULL, 'BBK', '운동화', NULL, 30,   2);

-- 정상적으로 실행
INSERT INTO buyTbl VALUES(NULL, 'JYP', '모니터', '전자', 200, 1);

-- 데이터 확인
SELECT * FROM userTbl;
SELECT * FROM buyTbl;

-- 테이블 정보 조회
DESCRIBE userTbl;

DROP TABLE IF EXISTS buyTbl, userTbl;

-- PRIMARY KEY를 지정하는 방법
-- 1. 테이블 생성시 해당 컬럼에 직접 지정
CREATE TABLE userTbl 
( userID  CHAR(8) NOT NULL PRIMARY KEY, 
  name    VARCHAR(10) NOT NULL, 
  birthYear   INT NOT NULL
);

DESCRIBE userTbl;

DROP TABLE IF EXISTS userTbl;

-- 2. 테이블 생성시 CONSTRAINT를 이용한 제약조건 지정
CREATE TABLE userTbl 
( userID  CHAR(8) NOT NULL, 
  name    VARCHAR(10) NOT NULL, 
  birthYear   INT NOT NULL,  
  CONSTRAINT PRIMARY KEY PK_userTbl_userID (userID)
);

DROP TABLE IF EXISTS userTbl;

-- 3. 테이블 생성 후 ALTER 문을 이용한 제약조건 지정
CREATE TABLE userTbl 
(   userID  CHAR(8) NOT NULL, 
    name    VARCHAR(10) NOT NULL, 
    birthYear   INT NOT NULL
);
ALTER TABLE userTbl
     ADD CONSTRAINT PK_userTbl_userID 
     PRIMARY KEY (userID);


DROP TABLE IF EXISTS prodTbl;

-- 다중 PRIMARY KEY 지정
CREATE TABLE prodTbl
( prodCode CHAR(3) NOT NULL,
  prodID   CHAR(4)  NOT NULL,
  prodDate DATETIME  NOT NULL,
  prodCur  CHAR(10) NULL
);
ALTER TABLE prodTbl
	ADD CONSTRAINT PK_prodTbl_proCode_prodID 
	PRIMARY KEY (prodCode, prodID) ;

DROP TABLE IF EXISTS prodTbl;
CREATE TABLE prodTbl
( prodCode CHAR(3) NOT NULL,
  prodID   CHAR(4)  NOT NULL,
  prodDate DATETIME  NOT NULL,
  prodCur  CHAR(10) NULL,
  CONSTRAINT PK_prodTbl_proCode_prodID 
	PRIMARY KEY (prodCode, prodID) 
);

SHOW INDEX FROM prodTbl;

DROP TABLE IF EXISTS buyTbl, userTbl;
CREATE TABLE userTbl 
( userID  CHAR(8) NOT NULL PRIMARY KEY, 
  name    VARCHAR(10) NOT NULL, 
  birthYear   INT NOT NULL 
);

-- FOREIGN KEY 지정 방법
-- 1. FORIEGN KEY로 직접 지정
CREATE TABLE buyTbl 
(  num INT AUTO_INCREMENT NOT NULL PRIMARY KEY , 
   userID  CHAR(8) NOT NULL, 
   prodName CHAR(6) NOT NULL,
   FOREIGN KEY(userID) REFERENCES userTbl(userID)
);

-- 2. CONSTRAINT를 이용해 지정
DROP TABLE IF EXISTS buyTbl;
CREATE TABLE buyTbl 
(  num INT AUTO_INCREMENT NOT NULL PRIMARY KEY , 
   userID  CHAR(8) NOT NULL, 
   prodName CHAR(6) NOT NULL,
   CONSTRAINT FK_userTbl_buyTbl FOREIGN KEY(userID) REFERENCES userTbl(userID)
);

-- 3. 테이블 생성 후 ALTER문을 이용해 지정
DROP TABLE IF EXISTS buyTbl;
CREATE TABLE buyTbl 
(  num INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
   userID  CHAR(8) NOT NULL, 
   prodName CHAR(6) NOT NULL 
);
ALTER TABLE buyTbl
    ADD CONSTRAINT FK_userTbl_buyTbl 
    FOREIGN KEY (userID) 
    REFERENCES userTbl(userID);

SHOW INDEX FROM buyTbl ;

ALTER TABLE buyTbl
	DROP FOREIGN KEY FK_userTbl_buyTbl; -- 외래 키 제거
ALTER TABLE buyTbl
	ADD CONSTRAINT FK_userTbl_buyTbl
	FOREIGN KEY (userID)
	REFERENCES userTbl (userID)
	ON UPDATE CASCADE;

-- UNIQUE 조건 지정
USE tabledb;
DROP TABLE IF EXISTS buyTbl, userTbl;
CREATE TABLE userTbl 
( userID  CHAR(8) NOT NULL PRIMARY KEY, 
  name    VARCHAR(10) NOT NULL, 
  birthYear   INT NOT NULL,  
  email   CHAR(30) NULL  UNIQUE
);
DROP TABLE IF EXISTS userTbl;
CREATE TABLE userTbl 
( userID  CHAR(8) NOT NULL PRIMARY KEY,
  name    VARCHAR(10) NOT NULL, 
  birthYear   INT NOT NULL,  
  email   CHAR(30) NULL ,  
  CONSTRAINT AK_email  UNIQUE (email)
);

-- CHECK 조건 지정
-- 출생연도가 1900년 이후 그리고 2023년 이전, 이름은 반드시 넣어야 함.
DROP TABLE IF EXISTS userTbl;
CREATE TABLE userTbl 
( userID  CHAR(8) PRIMARY KEY,
  name    VARCHAR(10) , 
  birthYear  INT CHECK  (birthYear >= 1900 AND birthYear <= 2023),
  mobile1	char(3) NULL, 
  CONSTRAINT CK_name CHECK ( name IS NOT NULL)  
);

-- 휴대폰 국번 체크
ALTER TABLE userTbl
	ADD CONSTRAINT CK_mobile1
	CHECK  (mobile1 IN ('010','011','016','017','018','019')) ;

-- DEFAULT 값 지정
DROP TABLE IF EXISTS userTbl;
CREATE TABLE userTbl 
( userID  	CHAR(8) NOT NULL PRIMARY KEY,  
  name    	VARCHAR(10) NOT NULL, 
  birthYear	INT NOT NULL DEFAULT -1,
  addr	  	CHAR(2) NOT NULL DEFAULT '서울',
  mobile1	CHAR(3) NULL, 
  mobile2	CHAR(8) NULL, 
  height	SMALLINT NULL DEFAULT 170, 
  mDate    	DATE NULL
);


DROP TABLE IF EXISTS userTbl;
CREATE TABLE userTbl 
( userID  	CHAR(8) NOT NULL PRIMARY KEY,  
  name    	VARCHAR(10) NOT NULL, 
  birthYear	INT NOT NULL,
  addr	  	CHAR(2) NOT NULL,
  mobile1	CHAR(3) NULL, 
  mobile2	CHAR(8) NULL, 
  height	SMALLINT NULL, 
  mDate    	DATE NULL
);
ALTER TABLE userTbl
	ALTER COLUMN birthYear SET DEFAULT -1;
ALTER TABLE userTbl
	ALTER COLUMN addr SET DEFAULT '서울';
ALTER TABLE userTbl
	ALTER COLUMN height SET DEFAULT 170;

-- default 문은 DEFAULT로 설정된 값을 자동 입력한다.
INSERT INTO userTbl VALUES ('LHL', '이혜리', default, default, '011', '1234567', default, '2023.12.12');
-- 열이름이 명시되지 않으면 DEFAULT로 설정된 값을 자동 입력한다
INSERT INTO userTbl(userID, name) VALUES('KAY', '김아영');
-- 값이 직접 명기되면 DEFAULT로 설정된 값은 무시된다.
INSERT INTO userTbl VALUES ('WB', '원빈', 1982, '대전', '019', '9876543', 176, '2020.5.5');
SELECT * FROM userTbl;
