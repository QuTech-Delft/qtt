from unittest import TestCase

import qcodes
import qcodes.tests.data_mocks

import qtt.data
#mport virtualDot

#mport virtualDot as msetup; from virtualDot import sample; 

#%%
class TestVirtualDot(TestCase):
    

    def setUp(self):
        pass
        #self.station = msetup.initialize(reinit=False, server_name=None, verbose=0)
        
    def tearDown(self):
        pass
        #msetup.close(verbose=0)

        
    def test_dotmodel(self):
        gate_map=msetup.DotModel.gate_map
        
        onedot=msetup.DotModel(name='dot', server_name=None, verbose=0)
        _=onedot.compute()

    def test_gates(self):
        pass
        #gates=self.station.gates

        #gates.set('L', 100.)
        #self.assertEqual( gates.get('L'), 100.)
        
    
if __name__=='__main__':
    t=TestVirtualDot()
    pass



