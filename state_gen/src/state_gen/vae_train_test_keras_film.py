import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import time
import state_gen_util as state_gen_util
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from tqdm import tqdm
import rospkg

ros_path = rospkg.RosPack()

fig = plt.figure()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


# train
""" Image data """
data = np.load(ros_path.get_path('vrep_jaco_data')+"/data/dummy_data.npy",allow_pickle=True)
train_dataset_list = []
test_dataset_list = []

for i in range(len(data)):
    if i % 10 == 0:
        test_dataset = []
        img = data[i][0][0]/5000
        img = np.reshape(img,[img.shape[0],img.shape[1],1])
        img = np.tile(img,(1,1,3))
        # plt.imshow(img)
        # plt.show()
        # plt.close()
        test_dataset.append(img)
        joint = data[i][1][0][:6]
        test_dataset.append(np.array(joint,dtype=np.float32))
        gp = data[i][1][0][6:]
        test_dataset.append(np.array(gp,dtype=np.float32))
        pressure = data[i][2][0]
        test_dataset.append(np.array(pressure,dtype=np.float32))
        test_dataset_list.append(test_dataset)
    else:
        train_dataset = []
        img = data[i][0][0]/5000
        img = np.reshape(img,[img.shape[0],img.shape[1],1])
        train_dataset.append(np.array(img,dtype=np.float32))
        joint = data[i][1][0][:6]
        train_dataset.append(np.array(joint,dtype=np.float32))
        gp = data[i][1][0][6:]
        train_dataset.append(np.array(gp,dtype=np.float32))
        pressure = data[i][2][0]
        train_dataset.append(np.array(pressure,dtype=np.float32))
        train_dataset_list.append(train_dataset)

""" training """
# train
epochs = 100
pic_data = []

train = True

if train:
    """ build graph """
    autoencoder = state_gen_util.Autoencoder(debug=False,isfushion=True)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder.compile(optimizer=optimizer,
                        loss = autoencoder.compute_loss)             
    autoencoder.fit(train_dataset_list,train_dataset_list,batch_size=20,epochs=100,verbose=True)
    # autoencoder.save_weights('weights/fushion_autoencoder_weights')
else:
    autoencoder_load = state_gen_util.Autoencoder(debug=False,isfushion=True)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    autoencoder_load.compile(optimizer=optimizer,
                        loss = autoencoder_load.compute_loss)
    autoencoder_load.train_on_batch(test_dataset_list,test_dataset_list)
    autoencoder_load.load_weights('weights/fushion_autoencoder_weights',)
    autoencoder_load.summary()
    
    def onChange(hey):
        pass

    eval_state_1 = np.zeros((1,32))
    sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
    sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))

    window_name = 'latent vector to img'
    cv.namedWindow(window_name)
    for i in range(16):
        tkbar_name = 'idx:'+str(i*2)
        cv.createTrackbar(tkbar_name,window_name, -30, 30, onChange)

    timechk = 0
    while True:
        timechk += 1
        cv.imshow(window_name, sampled_pic_1)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
        
        for i in range(int(len(eval_state_1[0])/2)):
            tkbar_name = 'idx:'+str(2*i)
            value = cv.getTrackbarPos(tkbar_name, window_name)/10
            eval_state_1[0][2*i] = value
            eval_state_1[0][2*i+1] = value
        tic = time.time()
        sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
        if timechk % 40 == 0:
            print("Sampled time = {:.5f}".format(time.time()-tic))
            timechk = 0
        sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))