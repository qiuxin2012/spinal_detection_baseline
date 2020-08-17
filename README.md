This branch is the implementation of [spinal_detection_baseline](https://github.com/wolaituodiban/spinal_detection_baseline) to run distributed PyTorch with [Analytics Zoo](https://github.com/intel-analytics/analytics-zoo).

**Running command:**
    ```bash
    python spinal_code/main.py --master spark://xxx:xxxx --worker_cores 4 --data [data path]
    ```

**Options:**
* `-d` `--data` This is required. The directory of the data.
* `-n` `--num_workers` The number of Horovod workers launched for distributed training. Default to be 1.
* `-c` `--worker_cores` The number of cores allocated for each worker. Default to be 4.
* `-e` `--epochs` The number of epochs to train the model. Default to be 20.
* `-b` `--batch_size` input batch size for training. Default to be 8.
* `-m` `--master` The master URL of an existing Spark standalone cluster. If not specified, a new standalone cluster would be started.
* `--use_bf16` Whether to use BF16 for model training if you are running on a server with BF16 support. Default to be False.
