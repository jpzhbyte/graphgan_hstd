import tensorflow as tf
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.io import loadmat
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.3

session = tf.Session(config=config)
bkg_img = loadmat('./Input/HYDICE/allbkg_cj.mat')
data_1 = bkg_img['allbkg_cj']
data = data_1

ano_img = loadmat('./Input/HYDICE/target_cj.mat')
data_a = ano_img['target_cj']
data_anom = data_a

d_img = loadmat('./Data/HYDICE_data.mat')
ori_d = d_img['d']
data_d = ori_d

input_dim = data.shape[1]
n_l1 = 500
n_l2 = 500
z_dim = 20

batch_size = data.shape[0]
batch_size_1 = data_anom.shape[0]
batch_size_2 = data_d.shape[0]
n_epochs = 2000
learning_rate = 1e-4
beta1 = 0.9

results_path = './Results/Spectral/'
path = './Results/Spectral/Saved_models/HYDICE/'
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
a_target = tf.placeholder(dtype=tf.float32, shape=[batch_size_1, input_dim], name='Anomaly')
d_target = tf.placeholder(dtype=tf.float32, shape=[batch_size_2, input_dim], name='d')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
real_distribution_1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Real_distribution_1')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')
x_vector = tf.placeholder(dtype=tf.float32, shape=[1, input_dim], name='xvector')
decoder_output_vector = tf.placeholder(dtype=tf.float32, shape=[1, input_dim], name='decodervector')
decoder_output = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Decoder_output')
def form_results():
    
    tensorboard_path = results_path  + '/Tensorboard'
    saved_model_path = results_path  + '/Saved_models/HYDICE/'
    log_path = results_path  + '/log/HYDICE/'
    feature_path = results_path  + '/feature'
    output_img_path = results_path  + '/output_img'
    encoder_weights_path = results_path  + '/encoder_weights'
    decoder_weights_path = results_path  + '/decoder_weights'
    encoder_bias_path = results_path  + '/encoder_bias'
    decoder_bias_path = results_path  + '/decoder_bias'
    if not os.path.exists(results_path ):
        os.mkdir(results_path )
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
        os.mkdir(feature_path)
        os.mkdir(output_img_path)
        

    return tensorboard_path, saved_model_path, log_path,feature_path,output_img_path


def generate_image_grid(sess, op):
    
    x_points = np.arange(-10, 10, 10).astype(np.float32)
    y_points = np.arange(-10, 10, 10).astype(np.float32)

    nx, ny = len(x_points), len(y_points)
    plt.subplot()   
    
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        
        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
        
        z = np.reshape(z, (1, 20))
        
        x = sess.run(op, feed_dict={decoder_input: z})
        
        ax = plt.subplot(g)
        
        img = np.array(x.tolist()).reshape(data_1.shape[0], data_1.shape[1])
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()


def dense(x, n1, n2, name):
    
   
    with tf.variable_scope(name, reuse=None):
        
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
    
        return out,weights,bias


def LeakyRelu(x, leak=0.2, name="LeakyRelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * tf.abs(x)
def encoder(x, reuse=False):
    if reuse:
        
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1,e_weights_1,e_bias_1 = dense(x, input_dim, n_l1, 'e_dense_1')
        e_dense_1 = LeakyRelu(e_dense_1)
        e_bias_1 = tf.reshape(e_bias_1,[1,n_l1])
        latent_variable,e_weights_2,e_bias_2 = dense(e_dense_1, n_l1, z_dim, 'e_latent_variable')
        e_weights = tf.matmul(e_weights_1,e_weights_2)
        cam_variance = global_variance_pooling(latent_variable)
        e_weights_2 = tf.sigmoid(e_weights_2)
        latent_variable_1 = tf.multiply(cam_variance,latent_variable)
        return latent_variable_1,e_weights_2


def decoder(x, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1,d_weights_1,d_bias_1 = dense(x, z_dim, n_l2, 'd_dense_1')
        d_bias_1 = tf.reshape(d_bias_1,[1,n_l2])
        d_dense_1 = LeakyRelu(d_dense_1)
        output, d_weights_2,d_bias_2= dense(d_dense_1, n_l2, input_dim, 'd_output')
        output = tf.nn.sigmoid(output)
        d_weights = tf.matmul(d_weights_1,d_weights_2)
        d_bias=tf.matmul(d_bias_1,d_weights_2)+d_bias_2
        return output,d_weights,d_bias


def discriminator(x, reuse=False):
    
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        dc_den1, _ , _ = dense(x, z_dim, n_l1, name='dc_den1')
        dc_den1 = LeakyRelu(dc_den1)
        output, _ , _ = dense(dc_den1, n_l1, 1, name='dc_output')
        
        return output

def discriminator_1(x, reuse=False):
    
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator_1'):
        dc_den1, _ , _ = dense(x, input_dim, n_l1, name='da_den1')
        dc_den1 = LeakyRelu(dc_den1)
        output, _ , _ = dense(dc_den1, n_l1, 1, name='da_output')
        return output


def global_variance_pooling(x):
	means,variance = tf.nn.moments(x, axes=[0,1], name='GvP')
	return variance


def train(train_model=True):
    defen = tf.Variable([0],name = 'defen',dtype=tf.float32)
    
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output, e_weights_output = encoder(x_input)
        decoder_output,d_weights,d_bias = decoder(encoder_output)

    with tf.variable_scope(tf.get_variable_scope()):
        d_real = discriminator(real_distribution)
        d_fake = discriminator(encoder_output, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        d_real_1 = discriminator_1(real_distribution_1)
        d_fake_1 = discriminator_1(decoder_output, reuse=True)

    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))-tf.reduce_mean(tf.square(tf.reduce_mean(a_target) - 0.025*tf.reduce_mean(decoder_output)))

    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))

    dc_loss = dc_loss_fake + dc_loss_real

    da_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_1), logits=d_real_1))
    da_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_1), logits=d_fake_1))

    da_loss = da_loss_fake + da_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()
    dc_var = [var for var in all_variables if 'dc_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]
    da_var = [var for var in all_variables if 'da_' in var.name]

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                    beta1=beta1).minimize(autoencoder_loss, var_list=en_var)                                               
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                    beta1=beta1).minimize(dc_loss, var_list=dc_var)
    discriminator_1_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                    beta1=beta1).minimize(da_loss, var_list=da_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                    beta1=beta1).minimize(generator_loss, var_list=en_var)

    init = tf.global_variables_initializer()
    # Saving the model
    saver = tf.train.Saver(max_to_keep=1000)
    step = 0
    with tf.Session() as sess:
        if train_model:
            tensorboard_path, saved_model_path, log_path,feature_path,output_img_path= form_results()
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            
            for i in range(n_epochs):
                n_batches = 1
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                for b in range(n_batches):
                    z_real_dist = np.random.randn(batch_size, z_dim) * 5.
                    
                    batch_xr = data
                    batch_x = data
                    batch_a = data_anom
                    batch_d = data_d
                    sess.run(autoencoder_optimizer,feed_dict={x_input: batch_x, x_target: batch_x, a_target: batch_a, d_target: batch_d})
                    sess.run(discriminator_optimizer,
                                feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                    sess.run(generator_optimizer,feed_dict={x_input: batch_x, x_target: batch_x})
                    e_output = sess.run(encoder_output,feed_dict={x_input: batch_x})
                    d_output = sess.run(decoder_output,feed_dict={x_input: batch_x})
                    sess.run(discriminator_1_optimizer,
                                feed_dict={x_input: d_output, real_distribution_1: batch_x})

                    if b % 1 == 0:

                        a_loss, d_loss, g_loss ,d_1_loss= sess.run(
                            [autoencoder_loss, dc_loss, generator_loss, da_loss],
                            feed_dict={x_input: batch_x, x_target: batch_x, a_target: batch_a, real_distribution: z_real_dist, real_distribution_1: batch_x, d_target: batch_d})
                    print("Epoch: {}, iteration: {}".format(i, b))
                    print("Autoencoder Loss: {}".format(a_loss))
                    print("Discriminator Loss: {}".format(d_loss))
                    print("Discriminator_1 Loss: {}".format(d_1_loss))
                    print("Generator Loss: {}".format(g_loss))
                    with open(log_path + '/log.txt', 'a') as log:
                        log.write("Epoch: {}, iteration: {}\n".format(i, b))
                        log.write("Autoencoder Loss: {}\n".format(a_loss))
                        log.write("Discriminator Loss: {}\n".format(d_loss))
                        log.write("Discriminator_1 Loss: {}\n".format(d_1_loss))
                        log.write("Generator Loss: {}\n".format(g_loss))
                        
                    np.set_printoptions(threshold=np.inf)
                    encoder_path = './Results/Spectral/Feature/HYDICE/'
                    decoder_path = './Results/Spectral/Output_img/HYDICE/'
                   

                step += 1
                if i % 100 == 0:
                    saver.save(sess, save_path=saved_model_path, global_step=i)
        else:
            # Get the latest results folder
            print('test')
            test_path = './Results/Test_out/HYDICE/'
            saver.restore(sess, save_path=tf.train.latest_checkpoint('./Results/Sspectral/Saved_models/HYDICE/'))
            output_y = sess.run(decoder_output,feed_dict={x_input: data})

            with open (test_path+'test_urban1.txt','a') as h:
                np.savetxt(test_path+'urban1_TD4.txt',output_y)
if __name__ == '__main__':
    train(train_model=True)