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
