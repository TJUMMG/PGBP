# Aggregate and Discriminate: Pseudo Clips-Guided Boundary Perception for Video Moment Retrieval

This is implementation for the paper "Aggregate and Discriminate: Pseudo Clips-Guided Boundary Perception for Video Moment Retrieval" (**TMM 2024**)

```shell
# preparing environment
bash conda.sh
```
## Introduction
Video moment retrieval (VMR) aims to localize a video segment in an untrimmed video, that is semantically relevant to a language query. The challenge of this task lies in effectively aligning the intricate and information-dense video modality with the succinctly summarized textual modality, and further localizing the starting and ending timestamps of the target moments. Previous works have attempted to achieve multi-granularity alignment of video and query in a coarse to fine manner, yet these efforts still fall short in addressing the inherent disparities in representation and information density between videos and queries, leading to modal misalignments. In this paper, we propose a progressive video moment retrieval framework, initially retrieving the most relevant and irrelevant video clips to the query as semantic guidance, thereby bridging the semantic gap between video modality and language modality. Futher more, we introduce a pseudo clips guided aggregation module to aggregate densely relevant moment clips closer together and propose a discriminative boundary-enhanced decoder with the guidance of pseudo clips to push the semantically confusing proposals away. Extensive experiments on the Charades-STA, ActivityNet Captions and TACoS datasets demonstrate that our method outperforms existing methods. 
<div align=center>
<img width="692" alt="image" src="https://github.com/user-attachments/assets/a0055195-2055-40aa-abdb-883c4b87ef95">
</div>

## Dataset Preparation
We use [VSLNet's](https://github.com/IsaacChanghau/VSLNet) data. The visual features can be download [here](https://app.box.com/s/h0sxa5klco6qve5ahnz50ly2nksmuedw), for CharadesSTA we use the "new" fold, and for TACoS we use the "old" fold, annotation and other details can be found [here](https://github.com/IsaacChanghau/VSLNet/tree/master/prepare)
and then modify the line 81~91 of "dataset/BaseDataset.py" to your own path.

## Quick Start
**Train**
```shell script
python main.py --cfg experiments/activitynet/PGBP.yaml --mode train
python main.py --cfg experiments/charades/PGBP.yaml --mode train
python main.py --cfg experiments/tacos/PGBP.yaml --mode train

python main.py --cfg experiments/charades_len/PGBP.yaml --mode train
python main.py --cfg experiments/charades_mom/PGBP.yaml --mode train
```
a new fold "results" are created.

## Citation
If you feel this project helpful to your research, please cite our work.
