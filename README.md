- 二分类：run_binary_guwen.sh
- 意象NER：run_ner_guwen.sh
- 单字NER：run_nerSingle_guwen.sh
- baseline：baseline\CCPM-baseline-main\run.sh

- guwen baseline: baseline\CCPM-baseline-main\run_guwen.sh

- 数据集共现bug: check_predict_ensemble_test/eval.ipynb
- ensemble: check_predict_ensemble_test/eavl.ipynb



| 方法            | 验证集acc | 测试集acc  |
| --------------- | --------- | ---------- |
| 共现            | 0.8768    |            |
| 意象NER         | 0.8320    |            |
| 二分类          | 0.8268    |            |
| guwen baseline  | 0.8830    |            |
| ensemble        | 0.9250    | **0.9085** |
| oracle ensemble | 0.9808    | -          |

> 所有测试集输出结果在`upload`文件夹中

