python batchsize/functions/sample_experiment.py --image_path "E:\covidx" --model resnet18 --sample_train_size full --sample_val_size full
Device: cuda
Setting up dataset...
Original samples: 67863
Valid AP/PA samples: 53691
Loaded 53691 samples for split 'train'
Class distribution: PA=20388, AP=33303
Original samples: 8473
Valid AP/PA samples: 4186
Loaded 4186 samples for split 'val'
Class distribution: PA=1275, AP=2911
Sample Train Size: 53691
Sample Val Size: 4186

--- Testing Batch Size: 8 ---
Success! Max Memory: 354.02 MB

--- Testing Batch Size: 16 ---
Success! Max Memory: 701.75 MB

--- Testing Batch Size: 32 ---
Success! Max Memory: 1047.68 MB

--- Testing Batch Size: 64 ---
Success! Max Memory: 1741.15 MB

--- Testing Batch Size: 128 ---
Success! Max Memory: 3110.33 MB

--- Testing Batch Size: 256 ---
Success! Max Memory: 5885.91 MB

--- Testing Batch Size: 512 ---
Success! Max Memory: 11374.89 MB

--- Testing Batch Size: 1024 ---
Success! Max Memory: 22345.22 MB

--- Testing Batch Size: 2048 ---
OOM Error for Batch Size 2048

--- Testing Batch Size: 4096 ---
OOM Error for Batch Size 4096

--- Testing Batch Size: 8192 ---
OOM Error for Batch Size 8192

--- Final Results ---
Batch Size 8: Success (Mem: 354.02 MB)
Batch Size 16: Success (Mem: 701.75 MB)
Batch Size 32: Success (Mem: 1047.68 MB)
Batch Size 64: Success (Mem: 1741.15 MB)
Batch Size 128: Success (Mem: 3110.33 MB)
Batch Size 256: Success (Mem: 5885.91 MB)
Batch Size 512: Success (Mem: 11374.89 MB)
Batch Size 1024: Success (Mem: 22345.22 MB)
Batch Size 2048: OOM
Batch Size 4096: OOM
Batch Size 8192: OOM