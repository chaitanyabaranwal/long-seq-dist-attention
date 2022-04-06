#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from pyroute2 import NDB

ndb = NDB(log='on')

for record in ndb.addresses.summary():
    record_dict = record._as_dict()

    if record_dict['ifname'] == 'ipogif0' and ':' not in record_dict['address']:
        print(record_dict['address'])
        break