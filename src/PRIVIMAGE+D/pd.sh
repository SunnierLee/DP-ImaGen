for i in `seq 0 99`
    do
        python process_dataset.py --worker_id $i --num_workers 100 &
    done