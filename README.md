# in_context_imitation_learning


## Installation:

-> Follow the README.md in rlbench_dataset_gen to install coppellia. After, install pytorch, pyg, ml_collections, viser, and `pip install -e .` inside rlbench_dataset_gen.


## Training examples


For platonic in-context:
```bash
python /home/davidknigge/Documents/GitHub/in_context_imitation_learning/scripts/train_platonic_transformer.py --dataset_path /media/davidknigge/hard-disk2/storage/robotics/rlbench_20tasks_100episodes_16_steps_preprocessed/temporal/ --task close_drawer --task press_switch --task close_fridge --device cuda --batch_size 48 --epochs 1000 --config.horizon=16 --config.model.transformer.use_cls_token=False --policy=in_context 
```


For DiT in-context:
```bash
python ~/in_context_imitation_learning/scripts/train_diffusion_transformer.py --dataset_path /home/dknigge/project_dir/data/robotics/temporal/ --task close_drawer --task press_switch --task close_fridge --device cuda --batch_size 64 --epochs 1000
```
