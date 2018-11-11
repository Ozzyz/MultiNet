#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the TensorDetect model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import scipy as scp

# configure logging
if 'TV_IS_DEV' in os.environ and os.environ['TV_IS_DEV']:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, stream=sys.stdout)
else:
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, stream=sys.stdout)

#import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, os.path.realpath('incl'))

import tensorvision.train as train
import tensorvision.utils as utils
import tensorvision.core as core

#import tensorflow_fcn

import time

#import random

flags.DEFINE_string('name', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('project', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('logdir', None,
                    'Append a name Tag to run.')

flags.DEFINE_string('hypes', None,
                    'File storing model parameters.')

tf.app.flags.DEFINE_boolean(
    'save', True, ('Whether to save the run. In case --nosave (default) '
                   'output will be saved to the folder TV_DIR_RUNS/debug, '
                   'hence it will get overwritten by further runs.'))


def run_united_evaluation(meta_hypes, subhypes, submodules, subgraph, tv_sess, step=0):
    logging.info('Running Evaluation Scripts.')
    #Limit GPU usage when running on shared environment
    summary_writer = tv_sess['writer']
    models = meta_hypes['model_list']
    sess = tv_sess['sess']

    n = 0

    py_smoothers = {}
    dict_smoothers = {}
    for model in models:
        py_smoothers[model] = train.MedianSmoother(5)
        dict_smoothers[model] = train.ExpoSmoother(0.95)

    for model in models:
        eval_dict, images = submodules[model]['eval'].evaluate(
            subhypes[model], sess,
            subgraph[model]['image_pl'],
            subgraph[model]['inf_out'])

        train._write_images_to_summary(images, summary_writer, step)

        if images is not None and len(images) > 0:

            name = str(n % 10) + '_' + images[0][0]
            image_dir = subhypes[model]['dirs']['image_dir']
            image_file = os.path.join(image_dir, name)
            scp.misc.imsave(image_file, images[0][1])
            n = n + 1

        logging.info("%s Evaluation Finished. Results" % model)

        logging.info('Raw Results:')
        utils.print_eval_dict(eval_dict, prefix='(raw)   ')
        train._write_eval_dict_to_summary(
            eval_dict, 'Evaluation/%s/raw' % model, summary_writer, step)

        logging.info('Smooth Results:')
        names, res = zip(*eval_dict)
        smoothed = py_smoothers[model].update_weights(res)
        eval_dict = zip(names, smoothed)
        utils.print_eval_dict(eval_dict, prefix='(smooth)')
        train._write_eval_dict_to_summary(eval_dict, 'Evaluation/%s/smoothed' % model, summary_writer, step)

        train._write_images_to_disk(meta_hypes, images, step)

    logging.info("Evaluation Finished. All results will be saved to:")
    logging.info(subhypes[model]['dirs']['output_dir'])

    # Reset timer
    start_time = time.time()


def _print_training_status(hypes, step, loss_values, start_time, lr):
    # Prepare printing
    duration = (time.time() - start_time) / int(utils.cfg.step_show)
    examples_per_sec = hypes['solver']['batch_size'] / duration
    sec_per_batch = float(duration)

    if len(loss_values.keys()) >= 2:
        info_str = ('Step {step}/{total_steps}: losses = ({loss_value1:.2f}, {loss_value2:.2f});'
                    ' lr = ({lr_value1:.2e}, {lr_value2:.2e}); ({sec_per_batch:.3f} sec)')
        losses = list(loss_values.values())
        print(losses)
        print(type(losses))
        lrs = list(lr.values())
        logging.info(info_str.format(step=step, total_steps=hypes['solver']['max_steps'],
                                     loss_value1=losses[0], loss_value2=losses[1],
                                     lr_value1=lrs[0], lr_value2=lrs[1],
                                     sec_per_batch=sec_per_batch))
    else:
        assert(False)


def build_training_graph(hypes, queue, modules, first_iter):
    """
    Build the tensorflow graph out of the model files.

    Parameters
    ----------
    hypes : dict
        Hyperparameters
    queue: tf.queue
        Data Queue
    modules : tuple
        The modules load in utils.

    Returns
    -------
    tuple
        (q, train_op, loss, eval_lists) where
        q is a dict with keys 'train' and 'val' which includes queues,
        train_op is a tensorflow op,
        loss is a float,
        eval_lists is a dict with keys 'train' and 'val'
    """

    data_input = modules['input']
    encoder = modules['arch']
    objective = modules['objective']
    optimizer = modules['solver']

    reuse = {True: False, False: True}[first_iter]

    scope = tf.get_variable_scope()

    with tf.variable_scope(scope, reuse=reuse):

        learning_rate = tf.placeholder(tf.float32)

        # Add Input Producers to the Graph
        with tf.name_scope("Inputs"):
            image, labels = data_input.inputs(hypes, queue, phase='train')

        # Run inference on the encoder network
        logits = encoder.inference(hypes, image, train=True)

    # Build decoder on top of the logits
    decoded_logits = objective.decoder(hypes, logits, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = objective.loss(hypes, decoded_logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        print(hypes)
        train_op = optimizer.training(hypes, losses, global_step, learning_rate)

    with tf.name_scope("Evaluation"):
        # Add the Op to compare the logits to the labels during evaluation.
        eval_list = objective.evaluation(hypes, image, labels, decoded_logits, losses, global_step)

        summary_op = tf.summary.merge_all()

    graph = {}
    graph['losses'] = losses
    graph['eval_list'] = eval_list
    graph['summary_op'] = summary_op
    graph['train_op'] = train_op
    graph['global_step'] = global_step
    graph['learning_rate'] = learning_rate

    return graph


def _recombine_2_losses(meta_hypes, subgraph, subhypes, submodules):
    if meta_hypes['loss_build']['recombine']:
        # Computing weight loss
        segmentation_loss = subgraph['segmentation']['losses']['xentropy']
        detection_loss = subgraph['detection']['losses']['loss']

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        if meta_hypes['loss_build']['weighted']:
            w = meta_hypes['loss_build']['weights']
            total_loss = segmentation_loss*w[0] + \
                detection_loss*w[1] + weight_loss
            subgraph['segmentation']['losses']['total_loss'] = total_loss
        else:
            total_loss = segmentation_loss + detection_loss + weight_loss
            subgraph['segmentation']['losses']['total_loss'] = total_loss

        for model in meta_hypes['model_list']:
            hypes = subhypes[model]
            modules = submodules[model]
            optimizer = modules['solver']
            gs = subgraph[model]['global_step']
            losses = subgraph[model]['losses']
            lr = subgraph[model]['learning_rate']
            subgraph[model]['train_op'] = optimizer.training(hypes, losses, gs, lr)


def _recombine_3_losses(meta_hypes, subgraph, subhypes, submodules):
    if meta_hypes['loss_build']['recombine']:
        # Read all losses
        segmentation_loss = subgraph['segmentation']['losses']['xentropy']
        detection_loss = subgraph['detection']['losses']['loss']
        road_loss = subgraph['road']['losses']['loss']

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        # compute total loss
        if meta_hypes['loss_build']['weighted']:
            w = meta_hypes['loss_build']['weights']
            # use weights
            total_loss = segmentation_loss*w[0] + \
                detection_loss*w[1] + road_loss*w[2] + weight_loss
        else:
            total_loss = segmentation_loss + detection_loss + road_loss \
                + weight_loss

        # Build train_ops using the new losses
        subgraph['segmentation']['losses']['total_loss'] = total_loss
        for model in meta_hypes['models']:
            hypes = subhypes[model]
            modules = submodules[model]
            optimizer = modules['solver']
            gs = subgraph[model]['global_step']
            losses = subgraph[model]['losses']
            lr = subgraph[model]['learning_rate']
            subgraph[model]['train_op'] = optimizer.training(hypes, losses, gs, lr)


def load_united_model(logdir):
    subhypes = {}
    subgraph = {}
    submodules = {}
    subqueues = {}

    subgraph['debug_ops'] = {}

    first_iter = True

    meta_hypes = utils.load_hypes_from_logdir(logdir, subdir="",
                                              base_path='hypes')
    for model in meta_hypes['model_list']:
        subhypes[model] = utils.load_hypes_from_logdir(logdir, subdir=model)
        hypes = subhypes[model]
        hypes['dirs']['output_dir'] = meta_hypes['dirs']['output_dir']
        hypes['dirs']['image_dir'] = meta_hypes['dirs']['image_dir']
        hypes['dirs']['data_dir'] = meta_hypes['dirs']['data_dir']
        submodules[model] = utils.load_modules_from_logdir(logdir, dirname=model, postfix=model)

        modules = submodules[model]

        logging.info("Build %s computation Graph.", model)
        with tf.name_scope("Queues_%s" % model):
            subqueues[model] = modules['input'].create_queues(hypes, 'train')

        logging.info('Building Model: %s' % model)

        subgraph[model] = build_training_graph(hypes, subqueues[model], modules, first_iter)

        first_iter = False

    if len(meta_hypes['model_list']) == 2:
        _recombine_2_losses(meta_hypes, subgraph, subhypes, submodules)
    else:
        _recombine_3_losses(meta_hypes, subgraph, subhypes, submodules)

    hypes = subhypes[meta_hypes['model_list'][0]]

    tv_sess = core.start_tv_session(hypes)
    sess = tv_sess['sess']
    saver = tv_sess['saver']

    cur_step = core.load_weights(logdir, sess, saver)
    for model in meta_hypes['model_list']:
        hypes = subhypes[model]
        modules = submodules[model]
        optimizer = modules['solver']

        with tf.name_scope('Validation_%s' % model):
            tf.get_variable_scope().reuse_variables()
            image_pl = tf.placeholder(tf.float32)
            image = tf.expand_dims(image_pl, 0)
            inf_out = core.build_inference_graph(hypes, modules, image=image)
            subgraph[model]['image_pl'] = image_pl
            subgraph[model]['inf_out'] = inf_out

        # Start the data load
        modules['input'].start_enqueuing_threads(hypes, subqueues[model], 'train', sess)

    target_file = os.path.join(meta_hypes['dirs']['output_dir'], 'hypes.json')
    with open(target_file, 'w') as outfile:
        json.dump(meta_hypes, outfile, indent=2, sort_keys=True)

    return meta_hypes, subhypes, submodules, subgraph, tv_sess, cur_step


def main(_):
    utils.set_gpus_to_use()

    load_weights = tf.app.flags.FLAGS.logdir is not None

    if not load_weights:
        print("You must specify --logdir path/to/trained/model")
        exit(1)

    utils.load_plugins()

    if 'TV_DIR_RUNS' in os.environ:
        os.environ['TV_DIR_RUNS'] = os.path.join(os.environ['TV_DIR_RUNS'], 'MultiNet')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        logdir = tf.app.flags.FLAGS.logdir
        logging_file = os.path.join(logdir, "output.log")
        utils.create_filewrite_handler(logging_file, mode='a')
        
        hypes, subhypes, submodules, subgraph, tv_sess, start_step = load_united_model(logdir)
        
        if start_step is None:
            start_step = 0

        # Run united training
        run_united_evaluation(hypes, subhypes, submodules, subgraph, tv_sess, step=start_step)

        # stopping input Threads
        tv_sess['coord'].request_stop()
        tv_sess['coord'].join(tv_sess['threads'])


if __name__ == '__main__':
    tf.app.run()
