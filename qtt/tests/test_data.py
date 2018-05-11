import unittest 

import qcodes
import qcodes.tests.data_mocks

import qtt.data

#%%


class TestData(unittest.TestCase):

    def test_transform(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()
        # print(dataset)
        tr = qtt.data.image_transform(dataset, arrayname='z')
        istep = tr.istep()
        self.assertEqual(istep, 1)
        # print(tr)

if __name__ == '__main__':
    t1 = TestData()
    t1.test_transform()
