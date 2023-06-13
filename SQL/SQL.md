# SQL 정리

## SQL문의 종류

- DQL(Data Query Language, 질의어): 테이블에 저장된 데이터를 검색할 때 사용하는 SQL문. SELECT가 대표적으로 해당

- DML(Data Manipulation Language, 데이터 조작어): 데이터베이스에 저장된 데이터를 조작하기 위해 사용하는 SQL문. INSERT(생성), UPDATE(수정), DELETE(삭제), MERGE(병합)이 해당한다

- TCL(Transaction Control Language, 트랜젝션 처리어): 트랜젝션과 관련된 작업을 처리하기 위한 SQL문. COMMIT, ROLLBACK이 해당된다.</br>- Transaction: 논리적으로 묶인 하나의 작업 단위. DML문에서 적용된다.

- DDL(Data Definition Language, 데이터 정의어): 객체를 생성, 수정, 삭제하는데 사용하는 SQL문. CREATE, ALTER, DROP이 해당한다. 자동으로 COMMIT되기 때문에 ROLLBACK으로 취소할 수 없고, 제거하기 위해서는 DROP문을 사용해야 한다.</br>- Oracle 객체: table, index, view, sequence, synonym

## 데이터의 유형

- 수치형

    - 정수: NUMBER(자릿수)
    
    - 실수: NUMBER(전체자릿수, 소수점자릿수)
    
- 문자형

    - 고정형 문자열: CHAR(byte)
    
    - 가변형 문자열: VARCHAR(byte)
    
    - 같은 byte수를 저장한다면 VARCHAR가 더 효율적임(CHAR는 무조건 지정한 바이트의 공간을 차지하기 때문)
    
## NULL

- 값이 없음을 의미한다.

- NULL이 포함된 연산은 결과값이 NULL을 가진다. 함수를 이용해서 NULL 값을 처리하여 연산할 수 있다.

- 기본적으로 컬럼은 NULL 값을 가질 수 있지만 강제적으로 NULL을 허용하지 않게 할 수 있다.

- NULL을 조회하려면 IS NULL 연산자를 사용한다.

- Oracle에서 NULL은 최댓값으로 인식한다.(MSSQL은 최솟값으로 인식)

## 제약조건

- 용도: 데이터 무결성을 위해 사용한다.

- Primary Key: 레코드를 식별하는 역할. Not NULL과 Unique 조건을 기본적으로 가지고 있으며, 테이블 당 1개만 존재 가능하다.

- Unique: 데이터의 값이 유일한 값을 가져야 하는 제약조건(중복 불가). 단, NULL은 허용한다.

- Not NULL: NULL 값을 허용하지 않는 제약조건.

- Check: 비즈니스 규칙을 지정하는 제약조건

- Foreign Key: 참조키. 두 개 이상의 테이블을 연결할 때 사용(join)