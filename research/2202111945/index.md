---
layout: post
type: research
date: 2022-02-11 19:45
category: Udacity
title: LS2. Accessing Published APIs
subtitle: "Designing RESTful APIs"
writer: 100
post-header: true  
header-img: ./img/2022-02-11-19-49-52.png
hash-tag: [restful, APIs]
use_math: true
toc : true
---

# LS2. Accessing Published APIs
> Designing RESTful APIs  
> Writer: SungwookLE    
> DATE: '22.2/11   

## 1. HTTP 이해하기
- `REST` 이해를 위해 `HTTP`를 알아보자 

1. `HTTP` *client-server* 플로우
![](./img/2022-02-11-19-49-52.png)
    - 클라이언트가 서버에 요청하면 Response로 다양한 형태의 데이터를 받을 수 있다.

2. `HTTP` requests는 아래와 같이 구성된다.
![](./img/2022-02-11-19-52-11.png)
    - 예시:
    ![](./img/2022-02-11-19-52-51.png)

3. `HTTP` Response
![](./img/2022-02-11-19-54-43.png)
    - 예시:
    ![](./img/2022-02-11-19-54-55.png)


## 2. 실습1: Request & Response using `postman`

- `postman`이라는 크롭 확장프로그램을 이용하여 간단하게 HTTP request를 보내고 response를 받아보자.

1. [코드](./img/api_server.py)를 `python` 으로 실행하여 localhost 서버를 열어두고,
![](./img/2022-02-12-17-06-57.png)

2. `postman`을 실행시켜 나의 서버 'url'에 method와 함께 요청을 보내면, response가 날라오는 것을 확인할 수 있다.
![](./img/2022-02-11-20-19-30.png)

- `HTTP`와 `CRUD` 사이의 관계

|HTTP|CRUD|
|----|----|
|GET|READ|
|POST|CREATE|
|PUT|UPDATE/CREATE|
|PATCH|UPDATE|
|DELETE|DELETE|

## 3. 실습2: google map API Request & Response using `postman`

- `google map API`를 이용하여 `request`와 `response`를 주고받자.

1. google cloud platform 가입
    - 가입을 해야, Google Maps API key를 준다.
2. 계정정보에서 `API키`를 저장해둔다.
![](./img/2022-02-11-20-52-04.png)
3. [공식 튜토리얼](https://developers.google.com/maps/documentation/geocoding/overview)을 따라 `Google Maps Geocoding API`를 사용한다.
4. `request` 보내고, `response` 받기
    - [해당 튜토리얼](https://developers.google.com/maps/documentation/geocoding/web-service-best-practices) 을 살펴보면, request [방법](./img/2022-02-11-20-53-28.png)이 나오는데, 아래의 코드를 복사하여 postman 프로그램에 입력하여 `request`를 보냈다.
    - 실행결과:
    ![](./img/2022-02-11-20-54-26.png)

5. 사용법보다 더 중요한건, google map API도 REST로 이루어졌다는 것인데, stateless한 url(`geocode/address=Los Angeles California, USA`)에 요청을 하면 `json` 형태로 response가 날라왔다.

## 4. 실습3: google map API Request & Response using `python`

- [구글맵API튜토리얼](https://developers.google.com/maps/)
- `python` 프로그램에선 아래와 같이 `http` request & response를 하면된다.
    - 패키지 `httplib2, json`을 이용 
    - google API url을 입력해주고, 생성한 `h`라는 http 객체에 `request`를 보내 return 값으로 `response`와 `content`를 받는다.
    - `content`는 `json` 포맷으로 변환하여 최종적으로 `python` 코드에 return한다.
    ![](./img/2022-02-12-11-25-37.png)
    - 실행결과
    ![](./img/2022-02-12-16-08-06.png)

## 5. 실습4: google map API 와 foursquares API 를 활용한 Mashup 프로젝트 in `python`

- 해당 실습은 코드만 분석하였음
    - `foursquare` API가 몇번 호출하면 회원가입시 받은 credit이 모두 소진되어 추가적으로 api를 사용하는데 돈을 내야하는 제한이 있습니다... ㅠ

- Input: `mealType`, `location`
- 프로젝트 구성: `mealType`과 `location`을 입력하면 location 주변의 mealType을 갖는 식당의 이름, 주소, 대표사진을 리턴해줌
- 코드: [솔루션코드](https://github.com/udacity/APIs/tree/master/Lesson_2/12_Make_Your_Own_Mashup/solution_code)

    1. google API를 이용하여 `inputString`을 입력하면 `latitude`, `longitude`를 리턴
    2. return 받은 `latitude`와 `longitude`를 기준으로 주변의 `mealType`의 식당을 검색
    3. 주변의 식당값이 return되면, return 값을 기준으로 값을 추출(사진 포함) 
    ![](./img/2022-02-12-16-58-07.png)  


## 끝
