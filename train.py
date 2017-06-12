import tensorflow as tf
import sugartensor as sg
from input_data import Surv
from model import ResNet

sg.sg_verbosity(10)
batch_size = 32

data = Surv(batch_size=32)

input_ = data.data
label = data.label

model = ResNet()
inference_fn = model.inference

# logit = model.inference(x)

def my_loss(input_, labels, inference_fn, num_gpu=1):
    assert num_gpu >= 0
    tower_loss = []

    input_batch = tf.split(input_, num_gpu, axis=0)
    label_batch = tf.split(labels, num_gpu, axis=0)

    for i in range(num_gpu):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('gpu_%d' % i):
                reuse = False if i == 0 else True
                logit = inference_fn(input_batch[i], reuse=reuse)

                loss = tf.reduce_mean(tf.square(logit - label_batch[i]))
                tower_loss.append(loss)

    return tower_loss

loss = tf.reduce_mean(logit.sg_ce(target=tf.cast(y, tf.int32)))

sg.sg_train(loss=my_loss(input_=input_, labels=label, inference_fn=inference_fn, num_gpu=4), lr=0.0001, ep_size=data.num_batch, log_interval=10, save_dir='log', optim='sgd')

# tf.sg_train(loss=my_loss(input_=input_, labels=label, inference_fn=inference_fn, num_gpu=4), ep_size=data.num_batch, max_ep=120)

