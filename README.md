# CoReCast: Component-Level Retrieval for Time Series Forecasting

CoReCast retrieves trend and seasonal components separately and integrates historical references into forecasting backbones. CRC-Linear combines CoReCast with a linear forecasting model.

![CoReCast introduction](assets/intro.png)

## Datasets



| Dataset | Channels | Input length L | Forecast horizon H |
| --- | ---: | ---: | ---: |
| Atec | 1569 | 1008 | 336  |
| Air Quality | 1105 | 84 | 28  |
| Temp | 3774 | 504 | 168  |
| Wind | 3696 | 672 | 168  |
| Traffic-CA | 7491 | 672 | 168  |
| Traffic-GBA | 2229 | 672 | 168  |
| Traffic-GLA | 3376 | 672 | 168  |
| Neurolib | 2000 | 1344 | 336  |
| SIRS | 2994 | 28 | 7  |
| Meter | 2898 | 1344 | 336  |
| SP500 | 1475 | 28 | 7  |
| Wiki-20k | 20000 | 28 | 7  |
| Electricity (ECL) | 321 | 512 | 96/192/336/720  |
| ETTh1 | 7 | 512 | 96/192/336/720  |
| ETTh2 | 7 | 512 | 96/192/336/720  |
| ETTm1 | 7 | 512 | 96/192/336/720  |
| ETTm2 | 7 | 512 | 96/192/336/720  |
| Weather | 21 | 512 | 96/192/336/720  |
| Traffic | 862 | 512 | 96/192/336/720  |
| Exchange | 8 | 512 | 96/192/336/720  |
| Illness | 7 | 512 | 24/36/48/60  |
| Solar | 137 | 512 | 96/192/336/720  |

Dataset sources: [U-CAST / Time-HD](https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series) and [TFB](https://github.com/decisionintelligence/TFB).

## Example: Atec + DLinear

Install PyTorch compatible with your CUDA environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the Atec data at `data/High_Dimensional_Time_Series/atec.csv`, then run the following command from the project root:

```bash
python -u scripts/run_component_raft.py \
  --data Atec --model DLinear --variant retrieval \
  --seq-len 1008 --pred-len 336 --freq t \
  --groups 157 --topm 10 --max-candidates 32 --candidate-stride 84 \
  --batch-size 32 --epochs 100 --patience 5 --learning-rate 0.001 \
  --seed 2021 --device cuda --output results/atec_dlinear
```

Results are saved in `results/atec_dlinear`.

## Experimental Results

The following tables are taken from Tables III, IV, and VI of the paper.

![Backbone comparison: MSE](assets/backbone_mse.png)

![Backbone comparison: MAE](assets/backbone_mae.png)

![CRC-Linear comparison](assets/crc_linear_highdim.png)

<details>
<summary>Additional experimental results from the paper</summary>

![Backbone comparison](assets/other_backbones.png)

![CRC-Linear comparison on other datasets](assets/other_model_results.png)

</details>

[LICENSE](LICENSE)
