# Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems
Liang Xin, Wen Song, Zhiguang Cao, Jie Zhang.  Multi-Decoder Attention Model with Embedding Glimpse for Solving Vehicle Routing Problems, Proceedings of the AAAI Conference on Artificial Intelligence, pp.12042-12049, 2021. [[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/17430)

To train the model (e.g. TSP with 20 nodes)
```bash
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20' --problem="tsp" --batch_size=512 --epoch_size=1280000 --kl_loss=0.01 --n_EG=2 --n_paths=5 --val_size=10000
```

To generate the test dataset for TSP20
```bash
mkdir data
python generate_data.py --filename=data/tsp_20 --problem="tsp" --graph_sizes=20 --dataset_size=100000
```

To evaluate the test dataset with pretrained model for TSP20
```bash
CUDA_VISIBLE_DEVICES="0" python -u search.py data/tsp_20.pkl --model=pretrained_tsp20/tsp_20/epoch-99.pt --beam_size=1 --eval_batch_size=1024
```

To Download all the pretrained models, generate the test datasets, evaluate with greedy strategy, 30-width and 50-width beam searching
```bash
bash eval.sh download
bash eval.sh generate_data
bash eval.sh tsp
bash eval.sh cvrp
bash eval.sh sdvrp
bash eval.sh op
bash eval.sh pctsp
bash eval.sh spctsp
```

Dependencies
* Python==3.10
* NumPy
* PyTorch==2.1.1 (Support with CUDA 12.3)
* tqdm
* tensorboard_logger

**Note:** The dependencies above has tested on CUDA 12.3 with fix error in some class `RuntimeError: __class__ not set defining 'StatePCTSP' as <class 'problems.pctsp.state_pctsp.StatePCTSP'>. Was __classcell__ propagated to type.__new__`

MDAM is developed based on https://github.com/wouterkool/attention-learn-to-route (Kool, W.; van Hoof, H.; and Welling, M. 2019. Attention,
Learn to Solve Routing Problems! In Proceedings of International Conference on Learning Representations (ICLR).)
