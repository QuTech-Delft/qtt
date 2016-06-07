import qcodes
import qcodes.tests.data_mocks

import qtt.data

#%%
def test_transform():
    dataset=qcodes.tests.data_mocks.DataSet2D()
    #print(dataset)
    tr = qtt.data.image_transform(dataset)
    #print(tr)

if __name__=='__main__':
    test_transform()



