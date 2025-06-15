### inference command
```shell
# late early model, early method
python opencood/rq_eval/rq2_inference.py --dataset_dir "/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_gen" --model_dir model/early_fusion --fusion_method early

```

### retrain command
```shell
python opencood/rq_eval/rq3_train.py --dataset_dir "/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_select" --model_dir model/late_fusion --method v2x_gen

```

### rq3 test
```shell
python opencood/rq_eval/rq3_inference.py --dataset_dir "/media/jlutripper/My Passport/v2x_dataset/rq3/rq3_test" --model_dir model/late_fusion --fusion_method late --scale 0.15

```


python opencood/rq2/rq3_inference.py --scale 0.1 --method ori --dataset_dir "/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_select/v2x_gen/0.15/early_fusion" --model_dir model/early_fusion --fusion_method early