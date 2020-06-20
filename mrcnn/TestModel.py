# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:39:37 2020

@author: meppenga
"""
import sys
sys.path.append(r"C:\MaskRCNN\TensorflowMaskRCNN\Mask_RCNN-master\mrcnn\UpgradeModel")
for p in sys.path:
    print(p)
from model import MaskRCNN 
# import mrcnn.model as modellib
from ConfigFileTraining import cellConfig
from CustumDataLoader import CellDataSetLoader
  # main issue now: data generator and laoding too much images (seems issue steps per epoch)
if __name__ == '__main__':
    
    config = cellConfig()
    config.UseRotation = False  
       
    dataset_train = CellDataSetLoader(config)
    # dataset_train.addCellImage(config.TrainDirectories, 'Training', config.UseSubFoldersTrain)
    dataset_train.addCellImage(r'M:\tnw\ist\do\projects\Neurophotonics\Brinkslab\Data\ML images\NewAnnotationDaanPart2\PMT\Critine\Octoscope2020-4-23 Archon library 400FOVs 4gridtrial_1', 'Training', config.UseSubFoldersTrain)
    # dataset_train.addSingleImage(r'C:\MaskRCNN\MaskRCNNGit\TensorFlow2MaskRCNN\MaskRCNNModel\TestData\Training\Round1_Coords31_R4950C0_PMT_0Zmax.png')
    dataset_train.prepare()    
    config.LogDir = r'C:\MaskRCNN\TensorflowMaskRCNN\Mask_RCNN-master\mrcnn\UpgradeModel\Training'
    weight_file = r'C:\MaskRCNN\TensorflowMaskRCNN\Mask_RCNN-master\mrcnn\UpgradeModel\WeightsMaskRCNNCell.h5'
        
    model = MaskRCNN(mode="training", model_dir=config.LogDir, config=config)
    model.load_weights(weight_file, by_name=True, exclude=config.ExludeWeights)
    model.train(dataset_train, dataset_train,
                        learning_rate=config.LEARNING_RATE,
                        epochs=3,
                        layers='all')