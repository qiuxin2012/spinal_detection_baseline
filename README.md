## Run example
- Run with FP32 datatype
    ```bash
    export PYTHONPATH=.
    python spinal_code/main.py --master spark://xxx:xxxx --worker_cores 4 --data [data path]
    ```
- Run with BF16 datatype
    ```bash
    export PYTHONPATH=.
    python spinal_code/main.py --master spark://xxx:xxxx --worker_cores 4 --data [data path] --use_bf16 True
    ```
