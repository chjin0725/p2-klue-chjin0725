# Pstage_03_KLUE_Relation_extraction

### training
* python train_my_bert.py --model_name=[model_name] --version=[version] --num_epochs=[number_of_epochs] --lr=[learning_rate] --batch_size=[batch_szie]
* ex) python train_my_bert.py --model_name=bert-base-multilingual-cased --version=_v1 --num_epochs=10 --lr=0.000025 --batch_size=16

### inference
* python inference_my_bert.py --model_dir=[model_path] --model_name=[model_name] --version=[version]
* ex) python inference.py --model_dir=./results/ --model_name=bert-base-multilingual-cased --version=_v1

