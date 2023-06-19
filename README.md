'''
python main.py --half=False --batch_size=128 --test_batch_size=128 \
    --step 90 100 --num_epoch=110 --n_heads=3 --num_worker=1 --k=1 \
    --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 \
    --use_vel=False --datacase=NTU60_CS --weight_decay=0.0005 \
    --num_person=1 --num_point=20 --graph=graph.ucla.Graph --feeder=feeders.feeder_ucla.Feeder
'''

