block_setting = dict()

#Conv block1 settings
conv_blk1 = dict()
conv_blk1['condition_method'] = 'conv-film'
conv_blk1['with_batch_norm'] = False
conv_blk1['n_features'] = 16
conv_blk1['kernel_size'] = 8
conv_blk1['stride'] = (2,2)
conv_blk1['padding'] = 'SAME'
conv_blk1['activation']='relu'

#Conv block2 settings
conv_blk2 = dict()
conv_blk2['condition_method'] = 'conv-film'
conv_blk2['with_batch_norm'] = False
conv_blk2['n_features'] = 32
conv_blk2['kernel_size'] = 4
conv_blk2['stride'] = (2,2)
conv_blk2['padding'] = 'SAME'
conv_blk2['activation']='relu'

#Conv block3 settings
conv_blk3 = dict()
conv_blk3['condition_method'] = 'conv-film'
conv_blk3['with_batch_norm'] = False
conv_blk3['n_features'] = 32
conv_blk3['kernel_size'] = 4
conv_blk3['stride'] = (2,2)
conv_blk3['padding'] = 'SAME'
conv_blk3['activation']='relu'

#Conv block4 settings
conv_blk4 = dict()
conv_blk4['condition_method'] = 'conv-film'
conv_blk4['with_batch_norm'] = False
conv_blk4['n_features'] = 64
conv_blk4['kernel_size'] = 3
conv_blk4['stride'] = (2,2)
conv_blk4['padding'] = 'SAME'
conv_blk4['activation']='relu'

#Conv block5 settings
conv_blk5 = dict()
conv_blk5['condition_method'] = 'conv-film'
conv_blk5['with_batch_norm'] = False
conv_blk5['n_features'] = 128
conv_blk5['kernel_size'] = 3
conv_blk5['stride'] = (2,2)
conv_blk5['padding'] = 'SAME'
conv_blk5['activation']='relu'

#Fushion layer settings
fushion = dict()
fushion['units'] = 256
fushion['survival_prob'] = 0.6

#FC layer1 settings
fc_1 = dict()
fc_1['units'] = 512
fc_1['survival_prob'] = 0.6

#FC layer2 settings
fc_2 = dict()
fc_2['units'] = 512
fc_2['survival_prob'] = 0.6

#Latent layer settings
latent_layer = dict()
latent_layer['survival_prob'] = 0.6

#DeFC layer1 settings
defc_1 = dict()
defc_1['units'] = 512
defc_1['survival_prob'] = 0.6

#DeFC layer2 settings
defc_2 = dict()
defc_2['units'] = 512
defc_2['survival_prob'] = 0.6

#Reshape layer settings
reshape = dict()
reshape['units'] = 15*20*conv_blk5['n_features']
reshape['survival_prob'] = 0.6

#DeConv block1 settings
deconv_blk1 = dict()
deconv_blk1['with_batch_norm'] = False
deconv_blk1['n_features'] = 64
deconv_blk1['kernel_size'] = 3
deconv_blk1['stride'] = (2,2)
deconv_blk1['padding'] = 'SAME'
deconv_blk1['activation']='relu'

#DeConv block2 settings
deconv_blk2 = dict()
deconv_blk2['with_batch_norm'] = False
deconv_blk2['n_features'] = 32
deconv_blk2['kernel_size'] = 3
deconv_blk2['stride'] = (2,2)
deconv_blk2['padding'] = 'SAME'
deconv_blk2['activation']='relu'

#DeConv block3 settings
deconv_blk3 = dict()
deconv_blk3['with_batch_norm'] = False
deconv_blk3['n_features'] = 32
deconv_blk3['kernel_size'] = 4
deconv_blk3['stride'] = (2,2)
deconv_blk3['padding'] = 'SAME'
deconv_blk3['activation']='relu'

#DeConv block4 settings
deconv_blk4 = dict()
deconv_blk4['with_batch_norm'] = False
deconv_blk4['n_features'] = 16
deconv_blk4['kernel_size'] = 4
deconv_blk4['stride'] = (2,2)
deconv_blk4['padding'] = 'SAME'
deconv_blk4['activation']='relu'

#DeConv block5 settings
deconv_blk5 = dict()
deconv_blk5['with_batch_norm'] = False
deconv_blk5['n_features'] = 1
deconv_blk5['kernel_size'] = 8
deconv_blk5['stride'] = (2,2)
deconv_blk5['padding'] = 'SAME'
deconv_blk5['activation']='relu'

#DeConv output settings
deconv_output = dict()
deconv_output['with_batch_norm'] = False
deconv_output['n_features'] = 1
deconv_output['kernel_size'] = 8
deconv_output['stride'] = (1,1)
deconv_output['padding'] = 'SAME'
deconv_output['activation']='relu'

#Setting Stack
block_setting['conv_block1'] = conv_blk1
block_setting['conv_block2'] = conv_blk2
block_setting['conv_block3'] = conv_blk3
block_setting['conv_block4'] = conv_blk4
block_setting['conv_block5'] = conv_blk5
block_setting['fushion'] = fushion
block_setting['fc_block1'] = fc_1
block_setting['fc_block2'] = fc_2
block_setting['latent_layer'] = latent_layer

block_setting['defc_block1'] = defc_1
block_setting['defc_block2'] = defc_2
block_setting['reshape'] = reshape
block_setting['deconv_block1'] = deconv_blk1
block_setting['deconv_block2'] = deconv_blk2
block_setting['deconv_block3'] = deconv_blk3
block_setting['deconv_block4'] = deconv_blk4
block_setting['deconv_block5'] = deconv_blk5
block_setting['deconv_output'] = deconv_output
