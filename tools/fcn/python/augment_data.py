import sys, os
# print sys.path
sys.path.insert(0,'/Users/robertsneddon/oirdsVIR/lib/python2.7/site-packages/PIL')
# print sys.path
import numpy as np
from PIL import Image
# from pillow import Image


def pca_augment(name):
    '''
    Given the name of an image, loads the raw data as an array
    name: the name of the image file (with path)
    pca_img: the new image created by pca

    '''
    imRaw = Image.open(name)
    img_arr = np.array(imRaw)
    covmat = cov_matrix(img_arr)
    evals, evecs = np.linalg.eig(covmat)
    pca_arr = augment_img(img_arr,evals,evecs)
    pca_arr = pca_arr.astype('uint8')
    pca_img = Image.fromarray(pca_arr)
    pca_img.save("/data/oirds/png/save/"+"test.png","png")
    return pca_img

def pca_augment_dif(name1,name2):
    '''
    Given the name of an image, loads the raw data as an array
    name: the name of the image file (with path)
    pca_img: the new image created by pca

    '''
    imRaw1 = Image.open(name1)
    img_arr1 = np.array(imRaw1)
    imRaw2 = Image.open(name2)
    img_arr2 = np.array(imRaw2)
    covmat = cov_matrix(img_arr1)
    evals, evecs = np.linalg.eig(covmat)
    pca_arr = augment_img(img_arr2,evals,evecs)
    pca_arr = pca_arr.astype('uint8')
    pca_img = Image.fromarray(pca_arr)
    pca_img.save("/data/oirds/png/save/"+"test.png","png")
    return pca_img


def cov_matrix(pic_arr):
    '''
    Args:
        pic_arr: the image in array format
    Returns: covmat, the covariance matrix
    '''
    # rgb_arr = pic_arr.flatten()
    rgb_arr = pic_arr.reshape(-1, pic_arr.shape[-1])
    rgb_arr=rgb_arr.astype(dtype=float)
    r=np.transpose(rgb_arr[:,0])
    g=np.transpose(rgb_arr[:,1])
    b=np.transpose(rgb_arr[:,2])
    # r=r-np.mean(r)
    # g=g-np.mean(g)
    # b=b-np.mean(b)
    # r=r.reshape((1,r.shape[0]))
    # g=g.reshape((1,g.shape[0]))
    # b=b.reshape((1,b.shape[0]))
    # covmat = np.cov(rgb_arr)
    covmat = np.cov((r,g,b),rowvar=1)
    return covmat

def augment_img(img_arr, evals, evecs):
    '''
        Args:
        img_arr: The image in numpy array format
        evals: The eigenvalues of the covariance matrix
        evecs: The eigenvectors of the covariance matrix

    Returns: aug_arr: The augmented image array

    '''
    clip = lambda x : 0 if x < 0 else 256 if x > 256 else x
    aug_arr = np.zeros((img_arr.shape[0],img_arr.shape[1],3),dtype=float)
    # norm_vec = np.random.normal(0,1.0,3)
    norm_vec = np.random.normal(0,0.1,size=None)*np.ones(3,"float")
    scale = [evals[k]*norm_vec[k] for k in range(0,len(norm_vec))]
    scale_vec = evecs.dot(scale)
    for i in range(0,img_arr.shape[0]):
        for j in range(0, img_arr.shape[1]):
            # norm_vec = np.random.normal(0,0.0025,3)
            # norm_vec = np.random.normal(0,0.005,size=None)*np.ones(3,"float")
            # norm_vec = np.random.normal(0,0.005,size=None)*np.ones(3,"float")
            # scale = [evals[k]*norm_vec[k] for k in range(0,len(norm_vec))]
            # scale_vec = evecs.dot(scale)
            aug_arr[i,j,:] = img_arr[i,j,:] + scale_vec
            # aug_arr[i,j,:] = map(clip, aug_arr[i,j,:])
    for k  in range(0,3):
        aug_arr[:,:,k]=(aug_arr[:,:,k] + 1.0 - np.min(aug_arr[:,:,k]))*(256/(np.max(aug_arr[:,:,k]) + 1.0 - np.min(aug_arr[:,:,k])))
    aug_arr=aug_arr.astype(dtype=int)
    return aug_arr

png_dir="/data/oirds/png/"
png_save_dir="/data/oirds/png/save/"
test_img="01195290_7681_257_7937_513.png"
test2_img="01195290_1_1_257_257.png"
test3_img="01195290_1_2817_257_3073.png"
test4_img="01195290_1_513_257_769.png"
test5_img="01195290_3073_2817_3329_3073.png"
name=png_save_dir+test_img
name2=png_save_dir+test2_img
name3=png_save_dir+test3_img
name4=png_save_dir+test4_img
name5=png_save_dir+test5_img
# pca_test=pca_augment(png_save_dir+test_img)