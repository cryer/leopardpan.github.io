
---
layout: post
title: ShadowSocks配置多用户
description: ShadowSocks配置多用户
---

# 步骤

* 利用putty，xshell或者其他的ssh软件登陆你的服务器端
* 以ubuntu系统为例：
```
cd /etc
vim shadowsocks.json
```
* 编辑成如下格式：其中port_password里是主要，包括端口和对应的密码，保存退出：wq。
```
{
"server":"0.0.0.0",
"local_address": "127.0.0.1",
"local_port":1080, 
"port_password":
{     "9999": "password1",
      "9998": "password2", 
      "9997": "password3",
      "9996": "password4"
},
"timeout":300,
"method":"aes-256-cfb",
"fast_open": false
}
```
* 后台启动ss服务,如果是云服务器的话，不需要设置开机自启，因为云服务器不会关机的。
```
ssserver -c /etc/shadowsocks.json -d start
```
关闭ss服务
```
ssserver -c /etc/shadowsocks.json -d stop
```
* 如果提示服务已在运行，它会给出pid号，运行
```
kill pid
```
然后再次运行上一步。
