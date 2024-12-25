SMITE
├── run.py                            -- script to train the models, runs factory.trainer.py                      
├── models                    
│   ├── unet.py                       -- inflated unet definition
|   ├── attention.py                  -- FullyFrameAttention to attend to all frames
|   ├── ...
├── src
|   ├── pipeline_smite.py             -- main pipeline containing all the important functions
|   ├── train.py                     
|   ├── inference.py                     
|   ├── slicing.py                    -- slices frames and latents for efficient attention processing across video sequences
|   ├── tracking.py                   -- tracker initialization, applies tracking to each frame, and uses feature voting
|   ├── frequency_filter.py           -- DCT filter for low-pass regularization
|   ├── metric.py                     
|   ├── latent_optimization.py        -- spatio-temporal guidance          
|   ├── ...   
├── scripts
|   ├── train.sh                      --script for model training
|   ├── inference.sh                  --script for model inference on videos
|   ├── test_on_images.sh             --script for testing the model on image datasets
|   ├── ...   
├── utils
|   ├── setup.py                     
|   ├── args.py                       -- define, parse, and update command-line arguments
|   ├── transfer_weights.py           -- transfer the 2D Unet weights to inflated Unet
|   ├── ...   