#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function,division,absolute_import

def strings_to_pbtxt(strings):
    for i, s in enumerate(strings):
        id_ = i + 1
        name = s.strip()
        print('''item {
  id: %s
  name: '%s'
}
''' % (id_,name))

if __name__ == '__main__':
    import sys
    with open(sys.argv[1],'r') as f:
        strings_to_pbtxt(f.readlines())
