# [LUWA Dataset: Learning Lithic Use-Wear Analysis on Microscopic Images](https://ai4ce.github.io/LUWA/).

[Jing Zhang](https://jingz6676.github.io//)\*, [Irving Fang](https://irvingf7.github.io/)\*,  [Hao Wu](https://www.linkedin.com/in/hao-wu-8bbb7724a/), [Akshat Kaushik](https://www.linkedin.com/in/akshat-kaushik/), [Alice Rodriguez](https://as.nyu.edu/departments/anthropology/people/graduate-students/doctoral-students/alice-rodriguez.html), [Hanwen Zhao](https://www.linkedin.com/in/hanwen-zhao-2523a4104/), [Juexiao Zhang](https://juexzz.github.io/), [Zhuo Zheng](https://zhuozheng.top/), [Radu Iovita](https://wp.nyu.edu/faculty-iovita/), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ)

![](./figs/figure1_v2.jpg)
## Project Website
Please visit [our project website](https://ai4ce.github.io/EgoPAT3Dv2/) for more information.


## Environment Setup
The project was developed on `Python 3.11.5` and `PyTorch 2.1.1` with `CUDA 11.8.0` binaries. While you can refer to `requirements.txt` for more details, having PyTorch installed should be mostly enough for this project.

We utilized `PyTorch 2.1.1` to access some of its exclusive features such as `torch.compile()` to accelerate training as much as we could. However, most (if not all) of these techniques should not affect the inference accuracy, so you should be able to perfectly replicate our results without a matching PyTorch version.


## Dataset
Please visit our [Hugging Face repo](https://huggingface.co/datasets/ai4ce/LUWA/tree/main) to access the dataset. Please refer to `transfer_learning
/data_utils/data_tribology.py` for how to process them. 

1. we use integers to label stone that has been worked again certain material as follows:
    | Material    | Integer |
    |:-------------|:-------|
    | ANTLER      | 0     |
    | BEECHWOOD   | 1     |
    | BEFOREUSE   | 2     |
    | BONE        | 3     |
    | IVORY       | 4     |
    | SPRUCEWOOD  | 5     |
    | BARLEY      | 6     |
    | FERN        | 7     |
    | HORSETAIL   | 8     |

    Here, "BEFOREUSE" refers to a state where the stone is not polished with any material at all.
2. In the dataset, $256, 512, 865$ refers to the resolution of the images. 
    a. The images were originally taken at the resolution of $865 \times 865$. This corresponds to the 1 granularity in the paper
    b. $512$ corresponds to the 6 granularity in the paper.
    c. $256$ corresponds to the 24 granularity in the paper.

## Fully-Supervised Image Classification
To reproduce the results in the *Fully-Supervised Image Classification* section of the paper, please refer to the `transfer_learning` folder.

1. To train a specific deep learning model, please run the following command.
    ```
    python dl_supervised_pipeline.py \
    --resolution "$RESOLUTION" \
    --magnification "$MAGNIFICATION" \
    --modality "$MODALITY" \
    --model "$MODEL" \
    --pretrained $PRETRAINED \
    --frozen $FROZEN \
    --vote $VOTE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --start_lr $START_LR \
    --seed $SEED
    ```

    You can take a look at any given script in `transfer_learning
    /launch_scripts/` for reference.

2. If you want to replicate the SVM-based model, please instead run the `svm_pipeline.py` 

3. There are some helper functions located in `transfer_learning/experiments/collect_results.py` to gather the inference results in a more presentable and readable format. Although ironically the helper script itself is not super tidy.

## Few-Shot Image Classification
in the *Few-Shot Image Classification* section of the paper, please refer to the `fewshot_learning` folder.