import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorlayer as tl
import inputs as data
import c3d_biclstm as net 
import time
from datetime import datetime
import threading

seq_len = 32
batch_size = 5
# Tamanho do lote era 8, e mudei para 5 por questoes de memoria da GPU, em todos os arquivos treino teste e validacao

num_classes = 84
dataset_name = 'librasds'
#dataset_name = 'isogr'
model_prefix='./'

curtime = '%s' % datetime.now()
d = curtime.split(' ')[0]
t = curtime.split(' ')[1]
strtime = '%s%s%s-%s%s%s' %(d.split('-')[0],d.split('-')[1],d.split('-')[2], 
                            t.split(':')[0],t.split(':')[1],t.split(':')[2])

x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3], name='x')
y = tf.placeholder(tf.int32, shape=[batch_size, ], name='y')
  
sess = tf.InteractiveSession()

_,networks = net.c3d_biclstm(x, num_classes, False, False)
network_pred = tf.nn.softmax(networks.outputs)
network_y_op = tf.argmax(tf.nn.softmax(networks.outputs),1)
network_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(network_y_op, tf.int32), y), tf.float32))
networks_cost = tl.cost.cross_entropy(networks_y, y)
  
sess.run(tf.initialize_all_variables())

# RGB
testing_datalist = './dataset_splits/test_rgb_list.txt'
X_test,y_test = data.load_video_list(testing_datalist)
X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
y_test  = np.asarray(y_test, dtype=np.int32)
rgb_prediction = np.zeros((len(y_test),num_classes), dtype=np.float32)
load_params = tl.files.load_npz(name='%s/isogr_rgb_birnn_model_epoch_10.npz'%(model_prefix))
tl.files.assign_params(sess, load_params, networks)
#networks.print_params(True)
test_iterations = 0
print '%s: rgb testing' % datetime.now()
for X_indices, y_label_t in tl.iterate.minibatches(X_teidx, 
                                                   y_test, 
                                                   batch_size, 
                                                   shuffle=False):
  # Read data for each batch      
  image_path = []
  image_fcnt = []
  image_olen = []
  is_training = []
  for data_a in range(batch_size):
    X_index_a = X_indices[data_a]
    key_str = '%06d' % X_index_a
    image_path.append(X_test[key_str]['videopath'])
    image_fcnt.append(X_test[key_str]['framecnt'])
    image_olen.append(seq_len)
    is_training.append(False) # Testing
    image_info = zip(image_path,image_fcnt,image_olen,is_training)
  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
                                      data.prepare_isogr_rgb_data)
  feed_dict = {x: X_data_t, y: y_label_t}
  dp_dict = tl.utils.dict_to_one(networks.all_drop)
  feed_dict.update(dp_dict)
  predict_value,accu_value,predict_loss= sess.run([network_pred, network_accu], feed_dict=feed_dict)
  rgb_prediction[test_iterations*batch_size:(test_iterations+1)*batch_size,:]=predict_value
  print('\RGB - test loss: {}, test acc: {}\n'.format(predict_loss, accu_value))

  test_iterations = test_iterations + 1

# Depth
testing_datalist = './dataset_splits/test_depth_list.txt'
X_test,y_test = data.load_video_list(testing_datalist)
X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
y_test  = np.asarray(y_test, dtype=np.int32)
depth_prediction = np.zeros((len(y_test),num_classes), dtype=np.float32)
load_params = tl.files.load_npz(name='%s/isogr_depth_birnn_model_epoch_10.npz'%(model_prefix))
tl.files.assign_params(sess, load_params, networks)
#networks.print_params(True)
test_iterations = 0
print '%s: depth testing' % datetime.now()
for X_indices, y_label_t in tl.iterate.minibatches(X_teidx, 
                                                   y_test, 
                                                   batch_size, 
                                                   shuffle=False):
  # Read data for each batch      
  image_path = []
  image_fcnt = []
  image_olen = []
  is_training = []
  for data_a in range(batch_size):
    X_index_a = X_indices[data_a]
    key_str = '%06d' % X_index_a
    image_path.append(X_test[key_str]['videopath'])
    image_fcnt.append(X_test[key_str]['framecnt'])
    image_olen.append(seq_len)
    is_training.append(False) # Testing
    image_info = zip(image_path,image_fcnt,image_olen,is_training)
  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
                                      data.prepare_isogr_depth_data)
  feed_dict = {x: X_data_t, y: y_label_t}
  dp_dict = tl.utils.dict_to_one(networks.all_drop)
  feed_dict.update(dp_dict)
  predict_value,accu_value,predict_loss= sess.run([network_pred, network_accu], feed_dict=feed_dict)
  depth_prediction[test_iterations*batch_size:(test_iterations+1)*batch_size,:]=predict_value
  print('\DEPTH - test loss: {}, test acc: {}\n'.format(predict_loss, accu_value))

  test_iterations = test_iterations + 1

## Flow
#testing_datalist = './dataset_splits/test_flow_list.txt'
#X_test,y_test = data.load_video_list(testing_datalist)
#X_teidx = np.asarray(np.arange(0, len(y_test)), dtype=np.int32)
#y_test  = np.asarray(y_test, dtype=np.int32)
#flow_prediction = np.zeros((len(y_test),num_classes), dtype=np.float32)
#load_params = tl.files.load_npz(name='%s/isogr_flow_birnn_model_epoch_10.npz'%(model_prefix))
#tl.files.assign_params(sess, load_params, networks)
##networks.print_params(True)
#test_iterations = 0
#print '%s: flow testing' % datetime.now()
#for X_indices, y_label_t in tl.iterate.minibatches(X_teidx, 
#                                                   y_test, 
#                                                   batch_size, 
#                                                   shuffle=False):
#  # Read data for each batch      
#  image_path = []
#  image_fcnt = []
#  image_olen = []
#  is_training = []
#  for data_a in range(batch_size):
#    X_index_a = X_indices[data_a]
#    key_str = '%06d' % X_index_a
#    image_path.append(X_test[key_str]['videopath'])
#    image_fcnt.append(X_test[key_str]['framecnt'])
#    image_olen.append(seq_len)
#    is_training.append(False) # Testing
#    image_info = zip(image_path,image_fcnt,image_olen,is_training)
#  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
#                                      data.prepare_isogr_flow_data)
#  feed_dict = {x: X_data_t, y: y_label_t}
#  dp_dict = tl.utils.dict_to_one(networks.all_drop)
#  feed_dict.update(dp_dict)
#  predict_value,_ = sess.run([network_pred, network_accu], feed_dict=feed_dict)
#  flow_prediction[test_iterations*batch_size:(test_iterations+1)*batch_size,:]=predict_value
#  test_iterations = test_iterations + 1

fusion_prediction = rgb_prediction + depth_prediction # + flow_prediction
prediction_values = tf.argmax(fusion_prediction, 1)


video_list_d = './dataset_splits/test_depth_list.txt'
#video_list_f = './dataset_splits/test_flow_list.txt'
video_list_r = './dataset_splits/test_rgb_list.txt'

f1 = open(video_list_d, 'r')
f_lines_d = f1.readlines()
f1.close()
print '%s: Leu arquivos de Depth' % datetime.now()

#f2 = open(video_list_f, 'r')
#f_lines_f = f2.readlines()
#f2.close()
#print '%s: Leu arquivos de fluxo' % datetime.now()

# Retirar comentarios para habilitar RGB
f3 = open(video_list_r, 'r')
f_lines_r = f3.readlines()
f3.close() 
print '%s: Leu arquivos de RGB' % datetime.now()

f = open('test_prediction_depth.txt', 'w')
for idx, line in enumerate(f_lines_d):
  linetxt = '%s %s %d\n' %(line.split(' ')[0], line.split(' ')[1], prediction_values[idx].eval()+1)
  f.write(linetxt)
f.close()

#f = open('test_prediction_flow.txt', 'w')
#for idx, line in enumerate(f_lines_f):
#  linetxt = '%s %s %d\n' %(line.split(' ')[0], line.split(' ')[1], prediction_values[idx].eval()+1)
#  f.write(linetxt)
#f.close()

f = open('teste_prediction_rgb.txt', 'w')
for idx, line in enumerate(f_lines_r):
  linetxt = '%s %s %d\n' %(line.split(' ')[0], line.split(' ')[1], prediction_values[idx].eval()+1)
  f.write(linetxt)
f.close()

# Teste mais recente realizado: 27-07-2018 - Somente RGB e Depth 

# In the end, close TensorFlow session.
sess.close()
print '%s: Finalizou ' % datetime.now()
