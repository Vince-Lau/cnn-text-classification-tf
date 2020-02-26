tunnel download -fd='||' -rd='|-|' t_load_text_classify_sample text_classify_sample.txt;

nohup tensorboard --logdir ./runs/1582427310/summaries/ > /home/log/tensorboard.log 2>&1 &

scp -P 9001 /d/lauyu/software/开发/python-dev/Anaconda3-5.2.0-Linux-x86_64.sh root@14.116.205.22:/root/data-lyj
scp /root/data-lyj/Anaconda3-5.2.0-Linux-x86_64.sh root@192.168.1.140:/home/data-lyj

nohup python train.py > /home/log/textcnn.log 2>&1 &

python train.py
python eval.py --checkpoint_dir='./runs/1582427310/summaries/'