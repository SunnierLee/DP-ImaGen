for i in `seq 0 29`
    do
        python process_dataset.py --worker_id $i --num_workers 30 &
    done