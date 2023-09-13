import sys
import os

import json

import paho.mqtt.client as mqtt

# 足球比赛主题
football_match_topic = ''
# 篮球比赛主题
basketball_match_topic = ''

# LOL比赛主题
lol_match_topic = ''
# DOTA2比赛主题
dota2_match_topic = ''
# CS:GO比赛主题
csgo_match_topic = ''
# KOG比赛主题
kog_match_topic = ''

# 用户名
username = 'szcjsj'
# 密码
password = 'caca52010327626a39318fcd8ee44cba'
#password = 'Cc654321'

# 连接回调
def on_connect(c, userdata, flags, rc):
    # The value of rc indicates success or not:
    # 0: Connection successful
    # 1: Connection refused - incorrect protocol version
    # 2: Connection refused - invalid client identifier
    # 3: Connection refused - server unavailable
    # 4: Connection refused - bad username or password
    # 5: Connection refused - not authorised
    # 6-255: Currently unused.

    # 连接成功
    if rc == 0:
        # 订阅相关主题
        c.subscribe(football_match_topic)
        #c.subscribe(basketball_match_topic)
        #c.subscribe(lol_match_topic)
        #c.subscribe(dota2_match_topic)
        #c.subscribe(csgo_match_topic)
        #c.subscribe(kog_match_topic)
    elif rc in [4, 5]:
        print('验证失败,请确认用户名、密钥、授权ip是否正确,否则会认证失败')


# 消息回调
def on_message(c, userdata, msg):
    # 消息处理逻辑，具体格式请参考文档
    print(msg.topic)
    print(json.loads(msg.payload))


if __name__ == '__main__':
    # websocket协议
    client = mqtt.Client(transport='websockets')
    client.tls_set()
    client.username_pw_set(username=username, password=password)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect("s.sportnanoapi.com", 443)
    client.loop_forever()
