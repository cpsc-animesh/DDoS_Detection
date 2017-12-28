'''
Created on Dec 26, 2017

@author: Animesh
'''

import socket
import scapy

'''
HOST = 'google.ca'
PORT = 80
for i in range(50):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.send('GET / HTTP/1.1\r\nHost: google.com\r\n\r\n')
    data = s.recv(1024)
    s.close()
    print 'Received', repr(data)
    print("\n")'''
    
p = sniff(count=10,filter="icmp and ip host 4.2.2.1")