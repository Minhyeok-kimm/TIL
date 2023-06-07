-- 12. 함수
-- 1) 단일행 함수
-- (1) 단일행 함수 - 문자열
-- 첫글자만 대문자로 변경(나머지는 모두 소문자 처리, mariaDB는 지원하지 않음)
SELECT INITCAP('ORACLE SERVER') FROM dual;
SELECT deptno, INITCAP(dname), INITCAP(loc) FROM scott.dept;

-- 소문자 및 대문자로 변경
SELECT LOWER('MANAGER'), UPPER('manager') FROM dual;
SELECT empno, ename, deptno FROM scott.emp WHERE LOWER(ename) = 'smith';

-- 문자열 연결
SELECT CONCAT('ORACLE', 'SERVER') FROM dual;
SELECT empno, ename, job, CONCAT(ename, job) FROM scott.emp;

-- 왼쪽 및 오른쪽 문자열 추가
SELECT LPAD('MILLER', 10), RPAD('MILLER', 10) FROM dual;
SELECT LPAD('MILLER', 10, '*'), RPAD('MILLER', 10, '*') FROM dual;
SELECT ename, LPAD(ename, 15, '*') FROM scott.emp;

-- 부분열 반환
SELECT SUBSTR('000101-3234232', 8, 1) FROM dual; -- 8번째 문자 1개 추출
SELECT SUBSTR('000101-3234232', 8) FROM dual;

-- 문자열 길이(반환값은 수치)
SELECT LENGTH('000101-3234232') FROM dual;

-- 모든 함수는 중첩이 가능(중첩함수): 함수(함수())
SELECT LENGTH(SUBSTR('000101-3234232', 8)) FROM dual;

-- 문자열 치환
SELECT replace('JACK and JUE', 'J', 'BL') FROM dual;
SELECT replace('JACK and JUE', 'J') FROM dual;

-- 특정 문자의 위치 반환(MariaDB는 시작위치 지정 불가)
SELECT INSTR('MILLER', 'L', 1) FROM dual;
SELECT INSTR('MILLER', 'L', 1, 2) FROM dual;
SELECT INSTR('MILLER', 'L', 5) FROM dual; -- 찾는 값이 없을 경우 0이 출력

-- 왼쪽 또는 오른쪽, 양쪽 공백(또는 문자) 제거(MariaDB에서는 공백제거만 가능)
-- 특정 문자 삭제
SELECT LTRIM('MILLERM', 'M') FROM dual;
SELECT LTRIM('MMMMMILLERM', 'M') FROM dual;
SELECT RTRIM('MILLERM', 'M') FROM dual;
SELECT RTRIM('MILLERMMMMM', 'M') FROM dual; -- 삭제할 문자가 여러개가 붙어있는 경우 모두 삭제

-- 공백 제거
SELECT LTRIM('         MILLERM       ') FROM dual;
SELECT RTRIM('         MILLERM       ') FROM dual;

-- 왼쪽 문자/공백 제거, 오른쪽 문자/공백 제거, 양쪽 문자/공백 제거 가능
SELECT TRIM(LEADING 1 FROM 111234561111) FROM dual;
SELECT TRIM(TRAILING 1 FROM 111234561111) FROM dual;
SELECT TRIM(BOTH 1 FROM 111234561111) FROM dual;
SELECT TRIM('      MAR     ') FROM dual;

-- 문제: 보안 이슈로 인해 주민번호 뒷자리는 첫글자가 보이고 나머지는 *로 표시
SELECT '891202-1234567' FROM dual;
SELECT CONCAT(SUBSTR('891202-1234567', 1, 8), '******') FROM dual;
SELECT REPLACE('891202-1234567', SUBSTR('891202-1234567', 9), '******') FROM dual;

SELECT RPAD(SUBSTR('891202-1234567', 1, 8), 14, '*') FROM dual;
SELECT SUBSTR('891202-1234567', 1, 8) || '******' FROM dual;

-- (2) 단일행 함수 - 수치
-- 올림 (주어진 숫자보다 크거나 같은 최소 정수)
SELECT CEIL(10.1) FROM dual;

-- 내림값 ( 주어진 숫자보다 작거나 같은 최대 정수 )
select FLOOR(10.1) from dual;

-- 나머지 (SQL이 아닌 다른 프로그램 언어에서는 %가 나머지를 구하는 연산자)
select MOD(10,3) from dual;

-- 반올림(양수의 경우 소수점 자릿수, 음수의 경우 정수 자릿수)
select ROUND(456.789) from dual;
select ROUND(456.789, 2) from dual; -- 가장 많이 사용되는 형식
select ROUND(456.789, -1) from dual;
select ROUND(456.789, -2) from dual;

-- 절삭(버림, MariaDB는 truncate이고 자리수 필수로 지정)
select TRUNC(456.789) from dual;
select TRUNC(456.789, 2) from dual;
select TRUNC(456.789, -1) from dual;
select TRUNC(456.789, -2) from dual;

-- 부호 식별(양수는 1, 음수는 -1, 0은 0)
SELECT SIGN(100), SIGN(-10), SIGN(0) FROM dual;

-- (3) 단일행 함수 - 날짜(DBMS별로 차이가 많이 남)
-- 현재 날짜( MariaDB는 SYSDATE(), NOW(), CURRENT_DATE() )
SELECT SYSDATE, SYSTIMESTAMP FROM dual;

-- 오라클은 날짜 연산이 가능
SELECT SYSDATE, SYSDATE + 1, SYSDATE -1 FROM dual;

-- 두 날짜 사이의 월 수 계산( MariaDB는 DATEDIFF(계산할 날짜1, 계산할 날짜2) )
SELECT MONTHS_BETWEEN(SYSDATE+100, SYSDATE) FROM dual;
SELECT TRUNC(MONTHS_BETWEEN(SYSDATE+100, SYSDATE)) FROM dual;

-- 월을 날짜에 더하거나 빼기( MariaDB는 DATE_ADD(날짜, interval 0 (년월일시분초)), DATE_SUB() )
SELECT SYSDATE, ADD_MONTHS(SYSDATE, 1), ADD_MONTHS(SYSDATE, -1) FROM dual;

-- 명시된 날짜로부터 다음 요일에 대한 날짜를 반환( MariaDB는 없음 )
SELECT NEXT_DAY(SYSDATE, '화') FROM dual; -- 영어 사용 불가. 인코딩이 한글로 되어있기 때문(데이터베이스 NLS에서 확인)
SELECT NEXT_DAY(SYSDATE, '수요일') FROM dual;

-- 월의 마지막 날 반환
SELECT LAST_DAY(SYSDATE) FROM dual;

-- 날짜 반올림
SELECT SYSDATE, round(SYSDATE, 'YEAR'), ROUND(SYSDATE, 'MONTH') FROM dual;

-- 날짜 절삭
SELECT SYSDATE, TRUNC(SYSDATE, 'YEAR'), TRUNC(SYSDATE, 'MONTH') FROM dual;

-- 날짜에서 특정 날짜 정보만 뽑기(TO_CHAR 대체)
SELECT SYSDATE, EXTRACT(YEAR FROM SYSDATE),
		EXTRACT(MONTH FROM SYSDATE),
		EXTRACT(DAY FROM SYSDATE),
		EXTRACT(HOUR FROM SYSTIMESTAMP),
		EXTRACT(MINUTE FROM SYSTIMESTAMP) FROM dual;
	
-- (4) 단일행 함수 - 변환함수
-- 문자를 숫자로 변환 ( MariaDB는 CAST )
SELECT TO_NUMBER('100') + 100 FROM dual;
SELECT TO_NUMBER('1000') + 100 FROM dual;
SELECT TO_NUMBER('1,000', '999,999') + 100 FROM dual; -- 쉼표가 들어간 경우 숫자로 인식하지 못해 포맷 지정 필요

-- 숫자를 문자로 변환
select to_char(1000) from dual;
select to_char(1000, 'L999,999') from dual; -- L: NLS에 지정된 화폐 단위(Locale 정보)
select to_char(1000, '$999,999') from dual;
select to_char(100000000, '$999,999,999') from dual;

-- 문자를 날짜로 변환
-- 다음 5가지 날짜표현식은 오라클에서 자동으로 날짜로 인식이 된다.
select to_date('2023/05/23') from dual;
select to_date('20230523') from dual;
select to_date('2023-05-23') from dual;
select to_date('2023,05,23') FROM dual;
select to_date('2023.05.23') from dual;

SELECT TO_DATE('2023,05,23', 'YYYY,MM,DD') FROM dual;
select to_date('20230523124554') from dual; -- 에러 발생
alter session set NLS_DATE_FORMAT='YYYY/MM/dd HH:MI:SS'; -- 임시적으로 바뀜
select to_date('20230523124554','YYYYMMddHHMISS') from dual;

select to_date('2023년05월23') from dual; -- 에러 발생
select to_date('2023년05월23일', 'YYYY"년"MM"월"dd"일"') from dual;

-- 날짜를 문자로 변경
SELECT SYSDATE, TO_CHAR(SYSDATE) FROM dual;
SELECT SYSDATE, TO_CHAR(SYSDATE, 'YYYY/MM/DD (AM)') FROM dual;
SELECT SYSDATE, TO_CHAR(SYSDATE, 'YYYY/MM/DD (AM) HH DAY DY HH24:MI:SS') FROM dual; -- AM, PM은 동일


-- (5) 단일행 함수 - case 함수, DECODE 함수(오라클 의존적)
select empno, ename, sal, job,
      case job when 'ANALYST' then sal*1.1
               when 'CLERK' then sal*1.2
               when 'MANAGER' then sal*1.3
               when 'PRESIDENT' then sal*1.4
               when 'SALESMAN' then sal*1.5
      END "급여"
from scott.emp;

select empno, ename, sal,
       CASE when sal >=0 and sal <=1000 then 'E' 
            when sal >1000 and sal <=2000 then 'D'
            when sal >2000 and sal <=3000 then 'C'
            when sal >3000 and sal <=4000 then 'B'
            when sal >4000 and sal <=5000 then 'A'
      END "등급"
from scott.emp;

-- 실습(4장. 단일함수 ppt 참고)
-- 1. 사원테이블에서 입사일이 12월인 사원의 사번, 사원명, 입사일을 검색
SELECT empno, ename, hiredate FROM scott.EMP WHERE EXTRACT(MONTH FROM hiredate) = 12;
SELECT empno, ename, hiredate FROM scott.emp WHERE to_char(hiredate, 'MM') = 12;

-- 2. 다음과 같은 결과를 검색할 수 있는 SQL 문장을 작성하시오
SELECT empno, ename, LPAD(sal, 10, '*') FROM scott.emp;

-- 3. 다음과 같은 결과를 검색할 수 있는 SQL 문장을 작성하시오
SELECT empno, ename, TO_CHAR(hiredate, 'YYYY-MM-DD') AS 입사일 FROM scott.emp;

