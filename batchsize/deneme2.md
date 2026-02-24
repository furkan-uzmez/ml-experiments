python batchsize/functions/sample_experiment.py --image_path "E:\covidx" --model resnet18 --sample_train_size full --sample_val_size full --batch_sizes 1024
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

--- Testing Batch Size: 1024 ---
Success! Max Memory: 22117.54 MB

--- Final Results ---
Batch Size 1024: Success (Mem: 22117.54 MB)