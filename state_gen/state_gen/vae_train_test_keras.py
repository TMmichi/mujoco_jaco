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
use_fit = True

""" Image data """
# data = np.load("./dummy_data.npy",allow_pickle=True)
data = np.load(ros_path.get_path('vrep_jaco_data')+"/data/dummy_data.npy",allow_pickle=True)
train_dataset_list = []
if not use_fit:
    for j in range(20):
        train_dataset = []
        for i in range(20):
            img = data[i][0][0]/5000
            img = np.reshape(img,[img.shape[0],img.shape[1],1])
            train_dataset.append(img)
        train_dataset = np.array(train_dataset,dtype=np.float32)
        train_dataset_list.append(train_dataset)
    train_dataset_list = np.array(train_dataset_list)
    print(train_dataset_list.shape)
    test_x = train_dataset_list[-1]
else:
    train_dataset = []
    test_x = []
    print(len(data))
    for i in range(len(data)):
        if i % 10 == 0:
            img = data[i][0][0]/5000
            img = np.reshape(img,[img.shape[0],img.shape[1],1])
            test_x.append(img)
        else:
            img = data[i][0][0]/5000
            img = np.reshape(img,[img.shape[0],img.shape[1],1])
            train_dataset.append(img)
    train_dataset = np.array(train_dataset,dtype=np.float32)
    test_x = np.array(test_x,dtype=np.float32)


""" training """
# train
epochs = 100
pic_data = []

train = False
if not use_fit:
    if train:
        """ build graph """
        autoencoder = state_gen_util.Autoencoder(debug=False)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        for epoch in tqdm(range(1, epochs + 1),desc="Epoch Iter"):
            start_time = time.time()
            for train_x in train_dataset_list:
                loss_val = autoencoder.compute_apply_gradients(train_x, optimizer).numpy().mean()
            end_time = time.time()
            print(loss_val)
            if epoch % 10 == 0:
                predictions = autoencoder.sample(1).numpy()
                img = np.tile(predictions[0],(1,1,3))
                pic_data.append(img)
                weight_name = "weights/Autoencoder_" + str(epoch)
                autoencoder.save(weight_name,save_format='tf')
    else:
        autoencoder_load = tf.keras.models.load_model('weights/autoencoder_weights',)
        autoencoder_load.summary()
        eval_pic = autoencoder_load.sample(1).numpy()
        eval_pic = np.tile(eval_pic[0],(1,1,3))
        pic_data.append(eval_pic)
else:
    if train:
        """ build graph """
        autoencoder = state_gen_util.Autoencoder(debug=False)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        autoencoder.compile(optimizer=optimizer,
                            loss = autoencoder.compute_loss)
        autoencoder.fit(train_dataset,train_dataset,batch_size=20,epochs=100)
        #autoencoder.save('weights/autoencoder_weights',save_format='tf')
        autoencoder.save_weights('weights/autoencoder_weights2')
    else:
        autoencoder_load = state_gen_util.Autoencoder(debug=False)
        optimizer = tf.keras.optimizers.Adam(1e-4)
        autoencoder_load.compile(optimizer=optimizer,
                            loss = autoencoder_load.compute_loss)
        autoencoder_load.train_on_batch(test_x,test_x)
        autoencoder_load.load_weights('weights/autoencoder_weights2',)
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
        
        while True:
            cv.imshow(window_name, sampled_pic_1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            
            for i in range(int(len(eval_state_1[0])/2)):
                tkbar_name = 'idx:'+str(2*i)
                value = cv.getTrackbarPos(tkbar_name, window_name)/10
                eval_state_1[0][2*i] = value
                eval_state_1[0][2*i+1] = value
            
            sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
            sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))

        '''
        eval_pic = autoencoder_load.predict(test_x)        
        eval_pic = np.tile(eval_pic[0],(1,1,3))
        gt_pic = np.tile(test_x[0],(1,1,3))
        
        eval_states = autoencoder_load.state(test_x).numpy()
        eval_state_02 = eval_states[2] - 0.2
        eval_state_02 = np.reshape(eval_state_02,[1,eval_state_02.shape[0]])
        sampled_pic_02 = autoencoder_load.sample(1,eval_state_02).numpy()
        sampled_pic_02 = np.tile(sampled_pic_02[0],(1,1,3))
        gt_pic_02 = np.tile(test_x[2],(1,1,3))

        eval_state_05 = eval_states[5] - 0.5
        eval_state_05 = np.reshape(eval_state_05,[1,eval_state_05.shape[0]])
        sampled_pic_05 = autoencoder_load.sample(1,eval_state_05).numpy()
        sampled_pic_05 = np.tile(sampled_pic_05[0],(1,1,3))
        gt_pic_05 = np.tile(test_x[5],(1,1,3))

        idx = 11
        eval_state_1 = eval_states[idx]
        
        for i in range(len(eval_state_1)):
            #eval_state_1[i] += 2*np.random.randn(1)
            eval_state_1[i]  = 0.1*i-4
        eval_state_1 = np.reshape(eval_state_1,[1,eval_state_1.shape[0]])
        sampled_pic_1 = autoencoder_load.sample(1,eval_state_1).numpy()
        sampled_pic_1 = np.tile(sampled_pic_1[0],(1,1,3))
        gt_pic_1 = np.tile(test_x[idx],(1,1,3))
        
        ax11 = fig.add_subplot(4,2,1)
        ax11.imshow(eval_pic)
        ax12 = fig.add_subplot(4,2,2)
        ax12.imshow(gt_pic)
        ax21 = fig.add_subplot(4,2,3)
        ax21.imshow(sampled_pic_02)
        ax22 = fig.add_subplot(4,2,4)
        ax22.imshow(gt_pic_02)
        ax31 = fig.add_subplot(4,2,5)
        ax31.imshow(sampled_pic_05)
        ax32 = fig.add_subplot(4,2,6)
        ax32.imshow(gt_pic_05)
        ax41 = fig.add_subplot(4,2,7)
        ax41.imshow(sampled_pic_1)
        ax42 = fig.add_subplot(4,2,8)
        ax42.imshow(gt_pic_1)
        #plt.show()'''


