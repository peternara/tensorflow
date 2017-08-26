1. tensorflow tf.train.SummaryWriter is deprecated, instead use tf.summary.FileWriter

https://stackoverflow.com/questions/41482913/module-object-has-no-attribute-summarywriter

2. tensorboard display:
	sh exec cmd : tensorboard --log_dir='logs/'
	
	info : Starting TensorBoard b'54' at http://cloud:6006
	
	limit : open url in chrome. firefox not support

3. git push use http proxy contain package must small than 1MB in size, so use ssh or git config --global http.postBuffer 157286400 (later method can't work)

https://confluence.atlassian.com/bitbucketserverkb/git-push-fails-fatal-the-remote-end-hung-up-unexpectedly-779171796.html 

main reason: github can't exceed 100M:
remote: error: See http://git.io/iEPt8g for more information.
remote: error: File tf_ocr_nn/model.ckpt.meta is 157.51 MB; this exceeds GitHub's file size limit of 100.00 MB

4. How to change git config from http to ssh ?

use command as follows:
git remote set-url origin git@github.com:someaccount/someproject.git
use git remote -V to check.

5. ERROR:tensorflow:Exception in QueueRunner: Run call was cancelled ?

Everything works correctly and the problem happens at the very last stage when python tries to kill threads. To do this properly you should create a train.Coordinator and pass it to your queue_runner (no need to pass sess, as default session will be used.

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    // do your things
    coord.request_stop()
    coord.join(threads)

https://stackoverflow.com/questions/38678371/tensorflow-enqueue-operation-was-cancelled

6. W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.

<1> root cause: These are not error messages but mere warning messages
The best way to maximise TF performance (apart from writing good code !!), is to compile it from the sources.
<2> close warnings:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

or export TF_CPP_MIN_LOG_LEVEL=2
https://stackoverflow.com/questions/42463594/cpu-instructions-not-compiled-with-tensorflow



