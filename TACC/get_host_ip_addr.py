#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@file        : get_host_ip_addr
@description :
@time        : 2021/4/19 8:26 pm
@author      : Li Shenggui
@version     : 1.0
'''

from pyroute2 import NDB

ndb = NDB(log='on')

for record in ndb.addresses.summary():
    record_dict = record._as_dict()

    if record_dict['ifname'] == 'ib0' and ':' not in record_dict['address']:
        print(record_dict['address'])
        break

