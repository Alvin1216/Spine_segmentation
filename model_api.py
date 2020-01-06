import matplotlib.pyplot as plt
import numpy as np
import os, cv2, random ,sys
from os import listdir
from os.path import isfile, join
from scipy.ndimage import label, generate_binary_structure
from unet import unet_original_size



def image_loader():
    height = 1200
    weight = 512

    base_path = "./data/"
    #原圖
    original_all_image = []
    original_all_label = []

    #nn格式＋ndarray格式＋正規化
    normalized_all_image = []
    normalized_all_label = []

    for folder_name in listdir(base_path):
        image_folder_path = base_path + folder_name + "/image/"
        try:
            for image_file_name in listdir(image_folder_path):
                #image_file_name as same as label_file_name
                label_folder_path = base_path + folder_name + "/label/"
                print(image_folder_path + image_file_name)
                print(label_folder_path + image_file_name)
                one_image = cv2.imread((image_folder_path + image_file_name),cv2.IMREAD_GRAYSCALE)
                one_image = cv2.copyMakeBorder(one_image,0,0,6,6,cv2.BORDER_REPLICATE)
                one_label = cv2.imread((label_folder_path + image_file_name),cv2.IMREAD_GRAYSCALE)
                one_label = cv2.copyMakeBorder(one_label,0,0,6,6,cv2.BORDER_REPLICATE)

                original_all_image.append(one_image)
                original_all_label.append(one_label)
        except:
            print('not a folder or something wrong!')

    normalized_all_image = (np.array(original_all_image)/255).reshape(60,1200,512,1)
    normalized_all_label = (np.array(original_all_label)/255).reshape(60,1200,512,1)

    train_image = np.float32(normalized_all_image[0:40])
    train_label = np.float32(normalized_all_label[0:40])
    test_image = np.float32(normalized_all_image[40:60])
    test_label = np.float32(normalized_all_label[40:60])

    return train_image,train_label,test_image,test_label

def model_loader(path):
    # base_path = "./model/"
    # if mode == 'f12_f3':
    #     model_best = unet_original_size()
    #     model_best.load_weights(base_path+"unet_membrane_wce300_1200512_v2.hdf5")
    #     print('model load finished!')
    #     return model_best
    # elif mode == 'f13_f2':
    #     model_best = unet_original_size()
    #     model_best.load_weights(base_path+"unet_membrane_wce300_1200512_v2.hdf5")
    #     print('model load finished!')
    #     return model_best
    # elif mode == 'f23_f1':
    #     model_best = unet_original_size()
    #     model_best.load_weights(base_path+"unet_membrane_wce300_1200512_v2.hdf5")
    #     print('model load finished!')
    #     return model_best
    # else:
    #     print('wrong_model_name!')
    try:
      model_best = unet_original_size()
      model_best.load_weights(path)
      return model_best
    except:
      print('model has something wrong!')

#gt 16個頻道的圖 拿出來做diledted 膨脹之後當作第一塊的dice mask
def conculate_every_aspine_dice(original_label,predict_label):
  #seperated_aspine input must be a uni8(opencv format) picture!
  dice_list = []
  img_pre = np.array(predict_label.reshape(1200,512)*255, dtype = np.uint8)
  original_label = np.array(original_label.reshape(1200,512)*255, dtype = np.uint8)
  label_seperate = seperated_aspine(original_label)
  kernel = np.ones((3,3),np.uint8)
  for one_aspine_label in label_seperate:
    one_bone_gt = np.array(one_aspine_label.reshape(1200,512)*255, dtype = np.uint8)
    dilation_mask = cv2.dilate(one_bone_gt,kernel,iterations = 4)
    one_bone_pre = cv2.bitwise_and(img_pre,dilation_mask)
    #plt.subplot(1,2,1)
    #plt.imshow(one_bone_gt)
    #plt.subplot(1,2,2)
    #plt.imshow(one_bone_pre)
    dice_list.append(round(dice_cof(one_bone_gt,one_bone_pre),3))
    #print('dc', round(dice_cof(one_bone_gt,one_bone_pre),3))
  return dice_list


def dice_cof(label_gt,predict):
  label_gt_abs = len(np.where(label_gt >0 )[0])
  predict_abs = len(np.where(predict > 0)[0])
  intersection = np.array(cv2.bitwise_and(label_gt,predict),dtype=np.uint8)
  intersection_abs = len(np.where(intersection > 0)[0])
  #print(label_gt_abs,predict_abs,intersection_abs)
  dice = (2*intersection_abs) / (label_gt_abs + predict_abs)
  return dice

def overlap(original_image,predict_label):
  img_ori = np.array(original_image.reshape(1200,512)*255, dtype = np.uint8)
  backtorgb = cv2.cvtColor(img_ori,cv2.COLOR_GRAY2RGB)

  img_over = np.array(predict_label.reshape(1200,512)*255, dtype = np.uint8)
  ret, binary = cv2.threshold(img_over.astype(np.uint8),0,255,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  overlap = np.zeros((1200,512,3), dtype = np.uint8)
  cv2.drawContours(overlap,contours,-1,(255,0,0),3)
  
  overlapping = cv2.addWeighted(backtorgb, 0.6, overlap, 0.4, 0)
  #plt.imshow(overlapping)
  return overlapping


def seperated_aspine(original_label):
  label_cat = []
  print('original_label',type(original_label))
  print('label_shape',original_label.shape)
  #original_label  = np.array(original_label, dtype = np.uint8)

  s = generate_binary_structure(2,2)
  labeled_array, num_features = label(original_label, structure=s)
  #plt.figure(figsize=(16,16))
  print('num_features:',num_features)
  for area_number in range(0,num_features):
    one = np.where(labeled_array == area_number+1,1,0)
    #plt.subplot(5,4,area_number+1)
    #plt.imshow(one)
    label_cat.append(one)
  return label_cat

def generate_background(original_label):
  background = np.where(original_label==1,0,1)
  plt.imshow(background,cmap = 'gray')
  return background


def image_predict(single_test_image,single_test_label,model):
    predict = model.predict(single_test_image.reshape(1,1200,512,1),batch_size=4)
    dice_list = conculate_every_aspine_dice(original_label = single_test_label,predict_label = predict)
    return_dice_list = [None]*20
    for index in range(0,len(dice_list)):
        return_dice_list[index] = dice_list[index]
        print('DC',str(index+1),':',dice_list[index])
    
    #return_dice_list = [0]*20
    #for i in 
    #return_dice_list = dice_list.copy()
    return_dice_list[19] = round(np.mean(np.array(dice_list)),3)

    plt.subplot(1,4,1)
    plt.imshow(single_test_image.reshape(1200,512))
    plt.axis('off')
    plt.title('original')
    plt.subplot(1,4,2)
    plt.imshow(single_test_label.reshape(1200,512))
    plt.axis('off')
    plt.title('label')
    plt.subplot(1,4,3)
    plt.imshow(predict.reshape(1200,512),cmap='gray')
    plt.axis('off')
    plt.title('predict label')
    plt.subplot(1,4,4)
    plt.imshow(overlap(single_test_image,predict))
    plt.axis('off')
    plt.title('overlap')
    plt.savefig('final_result.png')
    #plt.show()

    print('return_dice_list',len(return_dice_list))
    return return_dice_list

    
#predict_index = 14

def image_reader(path):
    print(path)
    try:
      image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
      image = cv2.copyMakeBorder(image,0,0,6,6,cv2.BORDER_REPLICATE)
    except:
      print("沒有這張喔!")
    
    if image.shape != (1200,512):
        print("wrong_input_size!")
    else:
        normalized_image = np.array(image)/255
        return normalized_image

# model_best = model_loader(mode='f12_f3')
# for i in range(0,100):
#   name = input('輸入你要哪一張：')
#   if name == 'exit':
#     sys.exit(0)
#   path_image = './data/f03/image/'+ name + '.png'
#   path_label = './data/f03/label/'+ name + '.png'
#   original_image = image_reader(path_image)
#   label_image = image_reader(path_label)
#   image_predict(original_image,label_image,model_best)
# #print(np.float32(label.reshape(1200,512)).dtype)
#plt.imshow(np.float32(label.reshape(1200,512)))
#plt.show()


