# Mask-R-CNN_VGG-obj

This is a repository for extracting the pig datasets from the Mask RCNN model and train it through a VGG Neural Network. The main purpose of this project is to generate classification if the pig is standing up or laying down from a top down view. Data augmentation will be generated through the extracted datasets of trained Mask RCNN model.
This version might not be clean and efficient but further updates will be procedured.
This project was generated from matterpost/Mask_RCNN,

  https://github.com/matterport/Mask_RCNN

you will need to download the pretrained coco dataset in order to train the Mask RCNN model first.
Go to the link below and dowload the file
  
  mask_rcnn_coco.h5

  https://github.com/matterport/Mask_RCNN/releases/tag/v2.0

"dataset/train" directory folder obtains pig images from a top down view
      these informations will be used in further steps to extract the information needed
      from the Mask RCNN model

Run the code in the given order as shown
1) pig.py
    Training the pig datasets through the Mask RCNN model
    for training this code have the matching version of tensorflow and keras
    install first the matching version of
      
      tensorflow == 1.3.0
      keras==2.0.8
      
    you can either install it through
    
      pip install tensorflow == 1.3.0
      pip install keras == 2.0.8
   
    through the cmd lines depending on the OS you are using
    
    Also change the variables of keepdims into keep_dims under the mrcnn/model.py file because the    matching tensorflow version has a different variable match. Once trained, i recommend to change back the tensorflow and keras to the latest version back in order to run the code mask_NN_storage.py as shown below.
    
2) mask_rcnn_output.py
    Extracting the rois informations from the Mask RCNN trained model
3) mask_n_RGB_generator.py
    Generating the data augmentation from the rois information extracted from mask_rcnn_output.py
4) all the augmented datasets will be generated under
    
    croped_mask_from_img
    
  folder, you would need to differenitate individually of which dataset matches to the right category under
  
    croped_mask_from_img/std_up
    croped_mask_from_img/lay_down
    
    in order to have the same amount of data of std_up and lay_down
    croped_mask_from_img/lay_down2 would be another file in case to save more pigs which are laying down
    croped_mask_from_img/unconf would be saving the pig data which are hard to differentiate if it is standing up or laying down
    
 You must differentiate individually the datasets with the same amount of croped_mask_from_img/std_up and croped_mask_from_img/lay_down
 
 5)class_data_aug.py (classifying the augmented datasets based on the portion divided of std_up, lay_down from the user)
 
 6)mask_NN_storage.py (generating the VGG model)
    Be sure to have the matching version of tensorflow and keras with changing back keep_dims variables to keepdims under mrcnn/model.py
 
 7)mask_model_test.py (testing other pig datasets to check their accuracy)
    This last code is not mandatory to run. You would need to generate other testing datasets under croped_mask_from_img/test_data
    in order to run it
    
This is just a first commit for this project and as explained ahead, further updates will be generated.
