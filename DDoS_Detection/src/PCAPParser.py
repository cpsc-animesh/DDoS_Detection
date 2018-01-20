import dpkt
import time
from os import walk
import os
import sys


# def parse(f):
#     x = 0
#     try:
#         pcap = dpkt.pcap.Reader(f)
#     except:
#         print "Invalid Header"
#         return
#     
#     for ts, buf in pcap:
#         try:
#             eth = dpkt.ethernet.Ethernet(buf)
#         except:
#             continue
#         if eth.type != 2048:
#             continue
#         try:
#             ip = eth.data
#         except:
#             continue
# 
#         if ip.p == 6:
#             if type(eth.data) == dpkt.ip.IP:
#                 tcp = ip.data
# 
# 
#                 if tcp.dport == 80:
#                     try:
#                         http = dpkt.http.Request(tcp.data)
#                         x = x+1
#                     except:
#                         continue
#     print("Hello World")
#     print x

import time
from scapy.all import rdpcap

def parse(f):
    x = 0
    pcap = rdpcap(f)
    
    for p in pcap:
        try:
            if p.haslayer(TCP) and p.getlayer(TCP).dport == 80 and p.haslayer(Raw):
                x = x + 1
        except:
            continue
    
    print x

if __name__ == '__main__':

    path = '/home/animesh/Documents'
    file = 'dump.pcap'
    start = time.time()
    current = os.path.join(path, file)
    print current
    f = open(current)
    parse(f)
    f.close()
    end = time.time()
    print (end - start)




