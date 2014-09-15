# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:32:34 2014

@author: bharadwaj
"""

import lxml.html
import urllib2
import collections
import hashlib
globalTagset = set([])

def getTagDistribution(givenRoot, globalTagset):
    tagCounts = collections.defaultdict(int)
    for childReference in givenRoot.iter():
        if type(childReference.tag) is str:
            tagCounts[childReference.tag] += 1
            globalTagset.add(childReference.tag)
    return tagCounts
    
html=open('PLDataset/fetchedPages/0.html')    
doc = lxml.html.fromstring(html)
#doc.make_links_absolute(sys.argv[1])
print doc
print globalTagset
tagDistribution = getTagDistribution(doc, globalTagset)
print tagDistribution