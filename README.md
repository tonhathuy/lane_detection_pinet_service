# lane_detection_pinet_service

### Build docker
        git clonehttps://github.com/tonhathuy/lane_detection_pinet_service.git
        cd lane_detection_pinet_service
        docker-compose up
      
### [Config](https://github.com/tonhathuy/lane_detection_pinet_service/blob/main/docker-compose.yml#L10)
        port 5005

### Service Response
#### HEADERS
        Content-Type: application/json
#### BODY
          {
                'code': '1000', 
                'status': rcode.code_1000, 
                'predicts': [x,y],
                'process_time': timeit.default_timer()-start_time,
                'return': ''
          }
### Service Request

- [Base64](./test/test_predict_base64.py)
- [Binary](./test/test_predict_binary.py)
- [Binary numpy](./test/test_predict_binary_numpy.py)
- [Multi Binary](./test/test_predict_multi_binary.py)
- [Multi Part](./test/test_predict_multipart.py)
