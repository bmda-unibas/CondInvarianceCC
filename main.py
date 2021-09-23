#  Authors: Maxim Samarin, Vitali Nesterov, Mario Wieser
#  Contact: maxim.samarin@unibas.ch
#  Date: 21.09.2021
#
#  Sample implementation of our model published in "Learning Conditional Invariance through Cycle Consistency"
#  at GCPR 2021.
#

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from scipy.stats import special_ortho_group

from utils import plot_input_data, plot_selected_dimensions

set_seed = 1234


def generate_artificial_data(num_samples = 10000, dim_original = 2, dim_target = 5, sigma_noise = 0.2, scaling_factors = (1,1,1)):

    # Fix y_target_dim directly
    y_target_dim = 3

    scaling_factors = np.array(scaling_factors, dtype=float)[:dim_original]

    # Sample data points x in [-1,1]^dim_original interval
    x = np.random.uniform(-1,1, (num_samples,dim_original))

    # Apply no rotation for property y
    # y = np.sum(x*x/scaling_factors, axis = 1)

    # Apply rotation of 45 degrees 
    # Has no effect when we consider circles / spheres
    rot_angel = np.pi/4

    if dim_original == 2:
        y = (x[:,0] * np.cos(rot_angel) + x[:,1] * np.sin(rot_angel))**2/scaling_factors[0] \
            + (x[:,0] * np.sin(rot_angel) - x[:,1] * np.cos(rot_angel))**2/scaling_factors[1]

    elif dim_original == 3:
        # Rotation only in X1X2-plane
        y = (x[:,0] * np.cos(rot_angel) + x[:,1] * np.sin(rot_angel))**2/scaling_factors[0] \
            + (x[:,0] * np.sin(rot_angel) - x[:,1] * np.cos(rot_angel))**2/scaling_factors[1] \
            + x[:,2]**2/scaling_factors[2]


    # Add noise
    y = y + np.random.normal(0,sigma_noise,num_samples)

    y = y.reshape(-1,1)

    # Add new dimensions with only zeros
    x_pad = np.pad(x, pad_width=((0,0),(0,dim_target-dim_original)), mode='constant')
    y_pad = np.pad(y, pad_width=((0,0),(0,y_target_dim-y.shape[-1])), mode='constant')

    # Fill padded dimensions with random numbers drawn form N(mu=0,sigma=0.01)
    x_pad[:,dim_original:] = np.random.normal(loc=0.0, scale=0.01, size=(x.shape[0], dim_target-dim_original))
    y_pad[:,y.shape[-1]:] = np.random.normal(loc=0.0, scale=0.01, size=(y.shape[0], y_target_dim-y.shape[-1]))

    # Perform random rotation with fixed seed!
    random_rot = special_ortho_group.rvs(dim=dim_target, random_state=set_seed)
    random_rot_y = special_ortho_group.rvs(dim=y_target_dim, random_state=set_seed)

    inv_map_x = random_rot.T
    inv_map_y = random_rot_y.T

    x_transform = np.matmul(x_pad, random_rot)
    y_transform = np.matmul(y_pad, random_rot_y)

    # plot_input_data(x,y,dim_original)

    return x, y, x_transform, y_transform, inv_map_x, inv_map_y


def next_batch(batch_size = 256, dim_original = 2, dim_target = 5, scaling_factors=(1,1,1)):

    return generate_artificial_data(num_samples=batch_size, dim_original = dim_original, dim_target = dim_target, scaling_factors=scaling_factors)


def model(mode = "train", num_runs = 1, num_samples = 2000, batch_size = 256, exp_suffix='',
          higher_dim_mapping = True, target_dim = 5, original_dim = 2, scaling_factors = (1,1,1),
          check_point_path=''):


    # Create the test data set
    if higher_dim_mapping:

        print("------> Mapping from", original_dim, "dims to", target_dim, "dims!")

        x_test_original, y_test_original, x_test, y_test, inv_map_x, inv_map_y = generate_artificial_data(num_samples=num_samples,
                                                                                                          dim_original=original_dim,
                                                                                                          dim_target=target_dim,
                                                                                                          scaling_factors=scaling_factors)

        exp_suffix = exp_suffix + '_target-dim-'+ str(target_dim)


    else:
        print("------> Using original", original_dim, "dims.")

        x_test, y_test, _, _, _, _ = generate_artificial_data(num_samples=num_samples,
                                                              dim_original = original_dim,
                                                              scaling_factors=scaling_factors)

        exp_suffix = exp_suffix + '_original-dim-'+ str(original_dim)


    for _ in range(num_runs):

        start_time = time.strftime('%m-%d-%Y_%H%M%S')

        tf.reset_default_graph()

        # Dimensions of X and Y
        px = x_test.shape[-1]
        py = y_test.shape[-1]

        # Latent dimensions in property subspace Z0
        pz_y = py

        # Total number of latent dimensions in Z = (Z0,Z1)
        pz = pz_y + px

        tf_X = tf.placeholder(tf.float32, [None, px])
        tf_Y = tf.placeholder(tf.float32, [None, py])

        lagMul = tf.placeholder(tf.float32, [1, 1], name='lagMul')
        train_flag = tf.placeholder(tf.float32,[1,1], name='train')

        ##########
        # Encoder
        e1 = tf.contrib.layers.fully_connected(tf_X, 256, activation_fn = tf.nn.relu, scope = "enc.1")
        e2 = tf.contrib.layers.fully_connected(e1, 256, activation_fn = tf.nn.relu, scope = "enc.1.1")
        mu = tf.contrib.layers.fully_connected(e2, pz, activation_fn = None, scope = "enc.2")

        # During training num_datapoints = batch_size
        # Different from batch_size during test as we use the full dataset
        num_datapoints = tf.shape(mu)[0]

        # Adding constant noise drawn from N(0,1) to learnt latent means
        eps = tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        eps = eps * train_flag
        z = tf.add(eps, mu)


        ##########
        # Sparsity layer

        one = tf.constant(1.0, tf.float32)

        # Parametrise sparsity layer, see eq. (6) in the paper
        f_ft = tf.matmul(tf.transpose(mu), mu) * (tf.divide(one, tf.cast(num_datapoints, tf.float32)))
        diag_f_ft = tf.diag_part(f_ft)

        # We add an additional variance of 10 in the last dimension
        diag_add_noise_np = np.ones(pz, dtype=np.float32)
        diag_add_noise_np[-1] = 10.0
        diag_add_noise = tf.constant(diag_add_noise_np)

        diag_f_ft = tf.add(diag_f_ft, diag_add_noise)
        ixz = 0.5 * tf.log(diag_f_ft)


        ##########
        # Sampling of new data points

        # Compute sampling variance in latent space
        var_all = tf.reduce_mean(mu * mu, 0)

        factor_sampling_limit = 2
        left_sampling_limit = -factor_sampling_limit * tf.sqrt(var_all + 1e-4)
        right_sampling_limit = factor_sampling_limit * tf.sqrt(var_all + 1e-4)

        # Sample uniformly new data points in the latent space
        z2 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit, dtype = tf.float32)

        # Uniform samples with additional Gaussian noise
        z2 = z2 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype = tf.float32)

        # Uniform samples with fixed z0 coordinates, five times
        z3 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit,
                               dtype=tf.float32)

        z3 = z3 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        z3 = tf.concat((z[:,0:pz_y], z3[:,pz_y:]), axis=-1)

        z4 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit,
                               dtype=tf.float32)
        z4 = z4 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        z4 = tf.concat((z[:,0:pz_y], z4[:,pz_y:]), axis=-1)


        z5 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit,
                               dtype=tf.float32)
        z5 = z5 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        z5 = tf.concat((z[:,0:pz_y], z5[:,pz_y:]), axis=-1)


        z6 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit,
                               dtype=tf.float32)
        z6 = z6 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        z6 = tf.concat((z[:,0:pz_y], z6[:,pz_y:]), axis=-1)

        z7 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit,
                               dtype=tf.float32)
        z7 = z7 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        z7 = tf.concat((z[:,0:pz_y], z7[:,pz_y:]), axis=-1)

        z8 = tf.random_uniform(tf.stack((num_datapoints, pz)), left_sampling_limit, right_sampling_limit,
                               dtype=tf.float32)
        z8 = z8 + tf.random_normal(tf.stack((num_datapoints, pz)), 0, 1.0, dtype=tf.float32)
        z8 = tf.concat((z[:,0:pz_y], z8[:,pz_y:]), axis=-1)

        # Concatenation of original data mapped to latent space (z),
        # fully uniformly sampled (z2),
        # and uniformly sampled in invariant space with fixed property (i.e. fixed z0)
        z_concat = tf.concat((z, z2, z3, z4, z5, z6, z7, z8), 0)
        bs_multiplier_sampling = 8


        ##########
        # Property Decoder
        z_property = z_concat[:, 0:pz_y]

        dec_y_concat = tf.contrib.layers.fully_connected(z_property, 512, activation_fn = tf.nn.relu, scope = "dec.y.1")
        dec_y_concat = tf.contrib.layers.fully_connected(dec_y_concat, py, activation_fn = None, scope = "dec.y.2")

        # dec_y_concat: includes the original input as well as sampled data points
        # dec: decoded properties of original data points
        dec_y = dec_y_concat[0:num_datapoints,:]


        ##########
        # Decoder for input X
        Z1_decZ0 = tf.concat((dec_y_concat, z_concat[:,pz_y:]), axis=-1)
        dec_x_concat = tf.contrib.layers.fully_connected(Z1_decZ0, 512, activation_fn = tf.nn.relu)
        dec_x_concat = tf.contrib.layers.fully_connected(dec_x_concat, 512, activation_fn = tf.nn.relu)
        dec_x_concat = tf.contrib.layers.fully_connected(dec_x_concat, px, activation_fn = None)

        dec_x = dec_x_concat[0:num_datapoints,:]


        ##########
        # Cycle step
        # Reuse encoder
        e_cycle_1 = tf.contrib.layers.fully_connected(dec_x_concat, 256, activation_fn = tf.nn.relu, scope = "enc.1", reuse = True)
        e_cycle_2 = tf.contrib.layers.fully_connected(e_cycle_1, 256, activation_fn = tf.nn.relu, scope = "enc.1.1", reuse = True)
        mu_cycle = tf.contrib.layers.fully_connected(e_cycle_2, pz, activation_fn = None, scope = "enc.2", reuse = True)

        mu_cycle_property = mu_cycle[:, 0:pz_y]
        cycle_eps = tf.random_normal(tf.stack((bs_multiplier_sampling*num_datapoints, pz_y)), 0, 1.0, dtype=tf.float32)

        z_cycle_property = mu_cycle_property + cycle_eps

        # Reuse property decoder
        cycle_dec_y_concat = tf.contrib.layers.fully_connected(z_cycle_property, 512, activation_fn = tf.nn.relu, scope = "dec.y.1", reuse = True)
        cycle_dec_y_concat = tf.contrib.layers.fully_connected(cycle_dec_y_concat, py, activation_fn = None, scope = "dec.y.2", reuse = True)  ## decoded property


        ##########
        # Losses

        # Cycle consistency loss for properties
        cl = tf.reduce_sum(tf.square(dec_y_concat - cycle_dec_y_concat), 1)
        cycle_loss = tf.reduce_mean(cl)

        cycle_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(dec_y_concat[num_datapoints:,:] - cycle_dec_y_concat[num_datapoints:,:]), 1))

        # Reconstruction and latent losses on original data points
        x_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(dec_x - tf_X) / (2.0), 1) )

        # Mean absolute error for X reconstruction
        x_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(dec_x - tf_X), 1))

        # Prediction mean squared error
        y_reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.square(dec_y - tf_Y),1))

        # Mean absolute error for property prediction
        y_mae = tf.reduce_mean(tf.reduce_sum(tf.abs(dec_y - tf_Y), 1))

        latent_loss = tf.reduce_sum(ixz)

        # Value which will be annealed during training and assigned to placeholder lagMul
        lagMul_value = 1.9

        full_loss = latent_loss + lagMul * (1.0 * x_reconstr_loss + 1.0 * y_reconstr_loss + 5.0 * cycle_loss)

        ##########

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(full_loss)

        init_op_l = tf.local_variables_initializer()
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=100000)

        check_point_name = "{}{}".format(start_time, exp_suffix)

        # Variables to store best reconstruction and predictions MAEs values
        best_x_mae_test = np.inf
        best_y_mae_test = np.inf

        if mode == "train":

            with tf.Session(config=config) as sess:
                sess.run(init_op)
                sess.run(init_op_l)

                for iter in range(300000):

                    if higher_dim_mapping:
                        _, _, x_batch, y_batch, _, _ = next_batch(batch_size=batch_size, dim_original=original_dim,
                                                                  dim_target=target_dim, scaling_factors=scaling_factors)
                    else:
                        x_batch, y_batch, _, _, _, _ = next_batch(batch_size=batch_size, dim_original=original_dim,
                                                                  scaling_factors=scaling_factors)

                    _, mu_out, latent_loss_out, recon_x_out, pred_y_out = sess.run(
                        [optimizer, mu, latent_loss, x_reconstr_loss, y_reconstr_loss],
                        feed_dict={tf_X: x_batch, tf_Y: y_batch, lagMul: np.asarray([[lagMul_value]]),
                                   train_flag: np.asarray([[1.0]])}
                    )

                    if iter % 1000 == 0 and iter > 2000:

                        x_mae_test_out, y_mae_test_out, latent_loss_test_out, cycle_loss_test_out, var_test_out, mu_test = sess.run(
                            [x_mae, y_mae, latent_loss, cycle_loss, var_all, mu],
                            feed_dict={tf_X: x_test, tf_Y: y_test,
                                       lagMul: np.asarray([[lagMul_value]]),
                                       train_flag: np.asarray([[0.0]])
                                       })

                        variance_test_out = np.diag(np.cov(mu_test, rowvar=False))

                        print("Iteration: {:d}\nvar_test_out/variance_test_out = {}".format(iter, var_test_out/variance_test_out))

                        z0dim = variance_test_out[0:pz_y]
                        z1dim = variance_test_out[pz_y:]

                        # print("Iteration: {:d},\tX MAE: {:.4f},\tY MAE: {:.4f},\tLatent loss: {:.4f},\tCycle loss: {}".format(
                        #     iter, x_mae_test_out, y_mae_test_out, latent_loss_test_out, cycle_loss_test_out)
                        # )

                        print("Iteration: {:d},\tX MAE: {:.4f},\tY MAE: {:.4f},\tLatent loss: {:.4f},\tCycle loss: {}, Selected Z0 dim.s: {:d}, Selected Z1 dim.s: {:d}".format(
                            iter, x_mae_test_out, y_mae_test_out, latent_loss_test_out, cycle_loss_test_out, z0dim[(z0dim > 1)].size, z1dim[(z1dim > 1)].size)
                        )

                        # Increase Lagrange multiplier
                        lagMul_value = lagMul_value * 1.03

                        if x_mae_test_out <= best_x_mae_test and y_mae_test_out <= best_y_mae_test and iter > 50000:
                            best_y_mae_test = y_mae_test_out
                            best_x_mae_test = x_mae_test_out
                            print("save model: Selected Z0 dim.s: {:d}, Selected Z1 dim.s: {:d}".format(
                                z0dim[(z0dim > 1)].size, z1dim[(z1dim > 1)].size
                            ))

                            save_path = saver.save(sess, "pretrained/{}.ckpt".format(check_point_name))

                            plot_selected_dimensions(variance=variance_test_out, stddev_noise=1,
                                                     pz_y=pz_y, pz=pz, iter=iter,
                                                     check_point_name=check_point_name, mode_suffix=mode)




        elif mode == "test":

            with tf.Session() as sess:

                # How many property values shall be fixed?
                if original_dim == 2:
                    num_property_samples = 10
                else:
                    num_property_samples = 5

                # Plotting: Specify axes interval [-lim_value,+lim_value]
                lim_value = 3


                saver.restore(sess, check_point_path)

                check_point_name = os.path.split(check_point_path)[-1][:-5]

                ##########
                # Obtain invariance result:
                mu_out, y_out, x_mae_test_out, y_mae_test_out = sess.run([mu, dec_y, x_mae, y_mae], feed_dict={tf_X: x_test, tf_Y: y_test,
                                                               lagMul: np.asarray([[lagMul_value]]),
                                                               train_flag: np.asarray([[0.0]])})

                var = np.diag(np.cov(mu_out, rowvar=False))

                plot_selected_dimensions(variance=var, stddev_noise=1,
                                         pz_y=pz_y, pz=pz,
                                         check_point_name=check_point_name, mode_suffix=mode)

                var_factor = 1.0

                fixed_z0 = mu_out[:,0:pz_y]
                sample_z1 = np.random.uniform(-var_factor * np.sqrt(var + 1e-4)[pz_y:],
                                              var_factor * np.sqrt(var + 1e-4)[pz_y:],
                                              (x_test.shape[0],(pz-pz_y))
                                              )

                mu_sample = np.hstack((fixed_z0, sample_z1))

                x_sample = sess.run(dec_x, feed_dict={tf_X: x_test, tf_Y: y_test, lagMul: np.asarray([[lagMul_value]]),
                                                   train_flag: np.asarray([[0]]), z: mu_sample})

                y_sample = sess.run(dec_y, feed_dict={tf_X: x_sample, tf_Y: y_test, lagMul: np.asarray([[lagMul_value]]),
                                                train_flag: np.asarray([[0]])})

                invariance_result = np.mean(np.sum(np.abs(y_out - y_sample), axis=1))

                print("\nCheck point: {}\nX_MAE = {:.4f}, Y_MAE = {:.4f} Invariance = {:.4f}".format(check_point_name, x_mae_test_out,
                                                                                                     y_mae_test_out, invariance_result))


                ##########
                # Plot traversal

                var_z0 = var[0:pz_y]
                var_z1 = var[pz_y:]

                sample_z = np.zeros(shape=(num_samples, pz))

                for index_z0, var_z0_direction in enumerate(var_z0):

                    if original_dim == 2:
                        fig, ax = plt.subplots(pz-pz_y+1, 1, figsize=(3.5,4*(pz-pz_y)))
                        ax = ax.reshape(-1,1)

                    elif original_dim == 3:
                        fig = plt.figure(figsize=(8,4*(pz-pz_y)))
                        ax = [fig.add_subplot(pz-pz_y+1, 1, i+1, projection='3d') for i in range((pz-pz_y+1))]
                        ax = np.array(ax).reshape(pz-pz_y+1, 1)

                    # Equidistant points in z0
                    t_min = min(mu_out[:, index_z0])
                    t_max = max(mu_out[:, index_z0])
                    sample_z0 = np.linspace(start=t_min, stop=t_max, num=num_property_samples)

                    for index_val_z0, val_z0 in enumerate(sample_z0):
                        sample_z[:,index_z0] = val_z0


                        for index_z1, var_z1_direction in enumerate(var_z1):
                            # Enumerate starts at 0, index of first z1 latent direction
                            # requires addition of pz_y, i.e. number of z0 latent dim.s
                            index_z1 = index_z1 + pz_y

                            # Equidistant points in z0
                            t_min = min(mu_out[:, index_z1])
                            t_max = max(mu_out[:, index_z1])

                            t_min = t_min*1.8
                            t_max = t_max*1.8

                            sample = np.linspace(start=t_min, stop=t_max, num=num_samples)

                            sample_z[:,index_z1] = sample

                            # Sampling in a fixed latent dimension of Z1 subspace at fixed Z0 coordinates
                            x_sample, y_sample = sess.run([dec_x, dec_y],
                                                          feed_dict={tf_X: x_test, tf_Y: y_test,
                                                                     lagMul: np.asarray([[lagMul_value]]),
                                                                     train_flag: np.asarray([[0]]),
                                                                     z: sample_z})

                            if higher_dim_mapping:
                                # Rotate data back to original orientation and remove
                                # padded dimensions to original data input dimensionality
                                x_sample = np.matmul(x_sample, inv_map_x)[:,:original_dim]

                                if y_sample.shape[-1] != 1:
                                    y_sample = np.matmul(y_sample, inv_map_y)[:,0]


                            if original_dim == 2:
                                ax[index_z1-pz_y,0].scatter(x_sample[:,0], x_sample[:,1], marker='.')

                                ax[index_z1-pz_y,0].set_xlim(left=-lim_value, right=lim_value)
                                ax[index_z1-pz_y,0].set_ylim(bottom=-lim_value, top=lim_value)


                            elif original_dim == 3:
                                ax[index_z1-pz_y,0].scatter(x_sample[:,0], x_sample[:,1], x_sample[:,2], marker='.')

                                ax[index_z1-pz_y,0].set_xlim(-lim_value, lim_value)
                                ax[index_z1-pz_y,0].set_ylim(-lim_value, lim_value)
                                ax[index_z1-pz_y,0].set_zlim(-lim_value, lim_value)


                            ax[index_z1-pz_y,0].set_title(r"Latent dim. {}: $\sigma$={:.3f}".format(index_z1+1, np.sqrt(var[index_z1])))

                            # Reset this direction again to 0 to independently sample latent dir.s
                            sample_z[:,index_z1] = 0


                        sample_z1 = np.random.uniform(-4 * np.sqrt(var + 1e-4)[pz_y:], 4 * np.sqrt(var + 1e-4)[pz_y:],(num_samples,(pz-pz_y)))
                        sample_z[:,pz_y:] = sample_z1

                        non_selected_dims = np.sqrt(var)<=1
                        sample_z[:,non_selected_dims] = 0


                        if original_dim == 3:
                            ax[2,0].view_init(elev=13., azim=-120)
                            ax[-2,0].view_init(elev=0., azim=-152)


                        # Samples in all selected latent dimension of Z1 subspace at fixed Z0 coordinates
                        x_sample, y_sample = sess.run([dec_x, dec_y],
                                                      feed_dict={tf_X: x_test, tf_Y: y_test,
                                                                 lagMul: np.asarray([[lagMul_value]]),
                                                                 train_flag: np.asarray([[0]]),
                                                                 z: sample_z})

                        if higher_dim_mapping:
                            x_sample = np.matmul(x_sample, inv_map_x)[:,:original_dim]

                            if y_sample.shape[-1] != 1:
                                y_sample = np.matmul(y_sample, inv_map_y)[:,0]


                        # Joint plot
                        if original_dim == 2:
                            ax[-1,0].scatter(x_sample[:,0], x_sample[:,1], marker='.')
                            plt.tight_layout(h_pad=2)

                        elif original_dim == 3:
                            ax[-1,0].scatter(x_sample[:,0], x_sample[:,1], x_sample[:,2], marker='.')
                            plt.tight_layout(h_pad=3, w_pad=3)


                        ax[-1,0].set_title(r"Sample selected $Z_1$ dim.")

                        ax[-1,0].tick_params(color='red', labelcolor='red')

                        if original_dim == 2:
                            for spine in ax[-1,0].spines.values():
                                spine.set_edgecolor('red')
                        elif original_dim == 3:
                            ax[-1,0].xaxis.pane.set_edgecolor('red')
                            ax[-1,0].yaxis.pane.set_edgecolor('red')
                            ax[-1,0].zaxis.pane.set_edgecolor('red')

                        # Reset everything to 0
                        sample_z = np.zeros(shape=(num_samples, pz))

                    ax[-1,0].set_xlim(-lim_value, lim_value)
                    ax[-1,0].set_ylim(-lim_value, lim_value)

                    if original_dim == 3:
                        ax[-1,0].set_zlim(-lim_value, lim_value)


                    plt.savefig("plots/{}_Z0-latent-dim-{}_latent-traversal.png".format(check_point_name, index_z0))
                    plt.clf()

        else:
            print(mode, "not implemented!")




if __name__ == '__main__':

    print("\nLearning Conditional Invariance through Cycle Consistency\n\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='Set train or test mode', default="train",
                        choices=["train","test"])
    parser.add_argument('--experiment', help='Set the experimental setting', default="ellipse",
                        choices=["ellipse","ellipsoid"])
    parser.add_argument('--pretrained', help='Set (relative) path to pretrained model (ending with .ckpt)',
                        default="None")
    parser.add_argument('--batch_size', help='Set batch size', type=int, default=256)
    parser.add_argument('--num_runs_train', help='Set separate training runs to perform', type=int, default=5)
    parser.add_argument('--num_samples_test', help='Set number of data points to be sampled for the traversal results',
                        type=int, default=2000)
    parser.add_argument('--scaling_factors', help='Set scaling factors which change the axes of the ellipse / ellipsoid.'
                                                  ' Please provide a string of 2 or 3 comma-separated values',
                        type=str, default="4,2,1")

    args = parser.parse_args()



    if args.pretrained == "None":
        # If you want to test a pretrained model, but do not want to specify path to checkpoint
        if args.experiment.startswith('ellipse'):
            args.pretrained = 'pretrained/CondInvCC_ellipse_rot-4-2_target-dim-5.ckpt'
        elif args.experiment.startswith('ellipsoid'):
            args.pretrained = 'pretrained/CondInvCC_ellipsoid_rot-4-2-1_target-dim-5.ckpt'
        else:
            print("Please check args.experiment", args.experiment, ", not covered yet!")


    exp_suffix = '_' + args.experiment

    # Split scaling_factors string into a list of individual values
    scaling_factors = [int(factor) for factor in args.scaling_factors.split(',')]

    # For convenience, we work with a list of length 3
    # While this is not the case, append ones to the end
    while len(scaling_factors) < 3:
        scaling_factors.append(1)


    if exp_suffix.startswith('_ellipse'):
        original_dim = 2
        target_dim = 5

        if scaling_factors != (1, 1, 1):
            exp_suffix = exp_suffix + '_rot-{}-{}'.format(scaling_factors[0], scaling_factors[1])
    elif exp_suffix.startswith('_ellipsoid'):
        original_dim = 3
        target_dim = 5

        if scaling_factors != (1, 1, 1):
            exp_suffix = exp_suffix + '_rot-{}-{}-{}'.format(scaling_factors[0], scaling_factors[1], scaling_factors[2])
    else:
        print("Please, check exp_suffix=", exp_suffix, ", not covered yet!")
        sys.exit()


    if args.mode == 'train':
        model(mode=args.mode, num_runs=args.num_runs_train, batch_size=args.batch_size, exp_suffix=exp_suffix, target_dim=target_dim,
              original_dim=original_dim, scaling_factors=scaling_factors)
    elif args.mode == 'test':
        model(mode=args.mode, num_samples=args.num_samples_test, batch_size=args.batch_size, target_dim=target_dim,
              original_dim=original_dim, scaling_factors=scaling_factors, check_point_path=args.pretrained)
    else:
        print("Please check the specified mode!")

    print(args.mode, "finished!")
