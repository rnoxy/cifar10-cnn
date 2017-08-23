#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test_loadDataset.py
# Author: Rafał Nowak <rafal.nowak@cs.uni.wroc.pl>

import unittest

class TestLoadDataset(unittest.TestCase):
    """Test load_CIFAR_dataset function from utils"""
    def test_certain_images(self):
        from myutils import load_CIFAR_dataset
        data_training, data_testing = load_CIFAR_dataset(shuffle=False)
        
        sample_id = 9
        self.assertTrue( (data_training[sample_id-1][0][0,0,:] == [134, 186, 223]).all() )
        sample_id = 19
        self.assertTrue( (data_training[sample_id-1][0][30,31,:] == [91, 75, 64]).all() )
        self.assertTrue( (data_testing[sample_id-1][0][30,31,:] == [61, 103, 125]).all() )
        self.assertEqual( data_testing[sample_id-1][1], 8 )

    def test_shuffling(self):
        from myutils import load_CIFAR_dataset
        data_training, data_testing = load_CIFAR_dataset()
        
        sample_id = 192
        x_training = data_training[sample_id][0][:,:]
        y_training = data_training[sample_id][1]
        
        sample_id = 190
        x_testing = data_testing[sample_id][0][:,:]
        y_testing = data_testing[sample_id][1]

        data_training, data_testing = load_CIFAR_dataset(shuffle=True)
        found = False
        for i in range(0,50000):
            if ( data_training[i][0][:,:] == x_training ).all():
                if found:
                    self.fail()
                else:
                    found = True
                    self.assertEqual( y_training , data_training[i][1] )
        
        found = False
        for i in range(0,10000):
            if ( data_testing[i][0][:,:] == x_testing ).all():
                if found:
                    self.fail()
                else:
                    found = True
                    self.assertEqual( y_testing , data_testing[i][1] )


if __name__ == '__main__':
    unittest.main()