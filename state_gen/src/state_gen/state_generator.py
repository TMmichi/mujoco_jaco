import tensorflow as tf
import numpy as np
from state_gen.state_gen_util import dataFusionGraph

state_size = 10

class State_generator:
    def __init__(self,**kwargs):
        self.state_size = state_size
        self.sess = kwargs['sess']
        self.input_holder, self.keep_prob, self.state = self._build_graph()

    def generate(self,data_buff):
        #data_buff -> [data@timestep][data_type][data,timestamp]
        depth_arm = data_buff[-1][0][0]/5000
        depth_bed = data_buff[-1][1][0]/5000
        image_arm = data_buff[-1][2][0]/5000
        image_bed = data_buff[-1][3][0]/5000
        depth_arm = np.reshape(depth_arm,[1,depth_arm.shape[0],depth_arm.shape[1],1])
        depth_bed = np.reshape(depth_bed,[1,depth_bed.shape[0],depth_bed.shape[1],1])
        image_arm = np.reshape(image_arm,[1,image_arm.shape[0],image_arm.shape[1],1])
        image_bed = np.reshape(image_bed,[1,image_bed.shape[0],image_bed.shape[1],1])
        gripper_data = np.array(data_buff[-1][4][0])
        jointstate_data = np.array(data_buff[-1][5][0])
        pressure_data = np.array(data_buff[-1][6][0])
        inputs = dict({
                    self.input_holder[0]:depth_arm, 
                    self.input_holder[1]:depth_bed,
                    self.input_holder[2]:image_arm,
                    self.input_holder[3]:image_bed,
                    self.input_holder[4]:gripper_data,
                    self.input_holder[5]:jointstate_data,
                    self.input_holder[6]:pressure_data,
                    self.keep_prob:1
                    })
        return self.sess.run((self.state),feed_dict=inputs)

    def _build_graph(self):
        input_placeholder = []
        depth_arm_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_depth_arm')
        input_placeholder.append(depth_arm_input)
        depth_bed_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_depth_bed')
        input_placeholder.append(depth_bed_input)
        image_arm_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_image_arm')
        input_placeholder.append(image_arm_input)
        image_bed_input = tf.compat.v1.placeholder(tf.float32, shape=[None, 480,640,1], name='input_image_arm')
        input_placeholder.append(image_bed_input)
        jointstate_data = tf.compat.v1.placeholder(tf.float32, shape=[None, 6], name='input_jointstate')
        input_placeholder.append(jointstate_data)
        gripper_data = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='input_gripper')
        input_placeholder.append(gripper_data)
        pressure_data = tf.compat.v1.placeholder(tf.float32, shape=[None, 10], name='input_pressure')
        input_placeholder.append(pressure_data)
        keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        #TODO: place input_placeholder into dataFusionGraph
        state = dataFusionGraph()

        return input_placeholder, keep_prob, state
    
    def get_state_shape(self):
        self.stateSize = 9
        stateShape = (self.stateSize,)
        return stateShape