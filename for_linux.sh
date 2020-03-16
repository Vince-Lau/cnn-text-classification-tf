ssh -oPort=9001 root@14.116.205.22
密码feimei123
然后再 ssh root@192.168.1.140
密码 fm123

tunnel download -fd='||' -rd='|-|' t_load_text_classify_sample text_classify_sample.txt;

scp -P 9001 /d/lauyu/software/开发/python-dev/Anaconda3-5.2.0-Linux-x86_64.sh root@14.116.205.22:/root/data-lyj
scp /root/data-lyj/Anaconda3-5.2.0-Linux-x86_64.sh root@192.168.1.140:/home/data-lyj

nohup python train.py --l2_reg_lambda=0.5 --num_epochs=10 > /home/log/20200226_over_random.log 2>&1 &

nohup python eval.py --checkpoint_dir='./runs/1582714402/checkpoints' > /home/log/20200226_over_random_eval.log 2>&1 &

nohup tensorboard --logdir ./runs/1582714402/summaries/ > /home/log/tensorboard.log 2>&1 &
http://14.116.205.22:6006/