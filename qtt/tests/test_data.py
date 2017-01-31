from unittest import TestCase

import qcodes
import qcodes.tests.data_mocks

import qtt.data

#%%
class Test1(TestCase):
    def test_transform(self):
        dataset=qcodes.tests.data_mocks.DataSet2D()
        #print(dataset)
        tr = qtt.data.image_transform(dataset, arrayname='z')
	istep = tr.istep()
        #print(tr)
    
if __name__=='__main__':
    t1=Test1()
    t1.test_transform()



