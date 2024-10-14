---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- mteb
model-index:
- name: stella-large-zh-v2
  results:
  - task:
      type: STS
    dataset:
      type: C-MTEB/AFQMC
      name: MTEB AFQMC
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 47.34436411023816
    - type: cos_sim_spearman
      value: 49.947084806624545
    - type: euclidean_pearson
      value: 48.128834319004824
    - type: euclidean_spearman
      value: 49.947064694876815
    - type: manhattan_pearson
      value: 48.083561270166484
    - type: manhattan_spearman
      value: 49.90207128584442
  - task:
      type: STS
    dataset:
      type: C-MTEB/ATEC
      name: MTEB ATEC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 50.97998570817664
    - type: cos_sim_spearman
      value: 53.11852606980578
    - type: euclidean_pearson
      value: 55.12610520736481
    - type: euclidean_spearman
      value: 53.11852832108405
    - type: manhattan_pearson
      value: 55.10299116717361
    - type: manhattan_spearman
      value: 53.11304196536268
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_reviews_multi
      name: MTEB AmazonReviewsClassification (zh)
      config: zh
      split: test
      revision: 1399c76144fd37290681b995c656ef9b2e06e26d
    metrics:
    - type: accuracy
      value: 40.81799999999999
    - type: f1
      value: 39.022194031906444
  - task:
      type: STS
    dataset:
      type: C-MTEB/BQ
      name: MTEB BQ
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 62.83544115057508
    - type: cos_sim_spearman
      value: 65.53509404838948
    - type: euclidean_pearson
      value: 64.08198144850084
    - type: euclidean_spearman
      value: 65.53509404760305
    - type: manhattan_pearson
      value: 64.08808420747272
    - type: manhattan_spearman
      value: 65.54907862648346
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/CLSClusteringP2P
      name: MTEB CLSClusteringP2P
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 39.95428546140963
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/CLSClusteringS2S
      name: MTEB CLSClusteringS2S
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 38.18454393512963
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/CMedQAv1-reranking
      name: MTEB CMedQAv1
      config: default
      split: test
      revision: None
    metrics:
    - type: map
      value: 85.4453602559479
    - type: mrr
      value: 88.1418253968254
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/CMedQAv2-reranking
      name: MTEB CMedQAv2
      config: default
      split: test
      revision: None
    metrics:
    - type: map
      value: 85.82731720256984
    - type: mrr
      value: 88.53230158730159
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/CmedqaRetrieval
      name: MTEB CmedqaRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 24.459
    - type: map_at_10
      value: 36.274
    - type: map_at_100
      value: 38.168
    - type: map_at_1000
      value: 38.292
    - type: map_at_3
      value: 32.356
    - type: map_at_5
      value: 34.499
    - type: mrr_at_1
      value: 37.584
    - type: mrr_at_10
      value: 45.323
    - type: mrr_at_100
      value: 46.361999999999995
    - type: mrr_at_1000
      value: 46.412
    - type: mrr_at_3
      value: 42.919000000000004
    - type: mrr_at_5
      value: 44.283
    - type: ndcg_at_1
      value: 37.584
    - type: ndcg_at_10
      value: 42.63
    - type: ndcg_at_100
      value: 50.114000000000004
    - type: ndcg_at_1000
      value: 52.312000000000005
    - type: ndcg_at_3
      value: 37.808
    - type: ndcg_at_5
      value: 39.711999999999996
    - type: precision_at_1
      value: 37.584
    - type: precision_at_10
      value: 9.51
    - type: precision_at_100
      value: 1.554
    - type: precision_at_1000
      value: 0.183
    - type: precision_at_3
      value: 21.505
    - type: precision_at_5
      value: 15.514
    - type: recall_at_1
      value: 24.459
    - type: recall_at_10
      value: 52.32
    - type: recall_at_100
      value: 83.423
    - type: recall_at_1000
      value: 98.247
    - type: recall_at_3
      value: 37.553
    - type: recall_at_5
      value: 43.712
  - task:
      type: PairClassification
    dataset:
      type: C-MTEB/CMNLI
      name: MTEB Cmnli
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_accuracy
      value: 77.7269993986771
    - type: cos_sim_ap
      value: 86.8488070512359
    - type: cos_sim_f1
      value: 79.32095490716179
    - type: cos_sim_precision
      value: 72.6107226107226
    - type: cos_sim_recall
      value: 87.39770867430443
    - type: dot_accuracy
      value: 77.7269993986771
    - type: dot_ap
      value: 86.84218333157476
    - type: dot_f1
      value: 79.32095490716179
    - type: dot_precision
      value: 72.6107226107226
    - type: dot_recall
      value: 87.39770867430443
    - type: euclidean_accuracy
      value: 77.7269993986771
    - type: euclidean_ap
      value: 86.84880910178296
    - type: euclidean_f1
      value: 79.32095490716179
    - type: euclidean_precision
      value: 72.6107226107226
    - type: euclidean_recall
      value: 87.39770867430443
    - type: manhattan_accuracy
      value: 77.82321106434155
    - type: manhattan_ap
      value: 86.8152244713786
    - type: manhattan_f1
      value: 79.43262411347519
    - type: manhattan_precision
      value: 72.5725338491296
    - type: manhattan_recall
      value: 87.72504091653029
    - type: max_accuracy
      value: 77.82321106434155
    - type: max_ap
      value: 86.84880910178296
    - type: max_f1
      value: 79.43262411347519
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/CovidRetrieval
      name: MTEB CovidRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 68.862
    - type: map_at_10
      value: 77.106
    - type: map_at_100
      value: 77.455
    - type: map_at_1000
      value: 77.459
    - type: map_at_3
      value: 75.457
    - type: map_at_5
      value: 76.254
    - type: mrr_at_1
      value: 69.125
    - type: mrr_at_10
      value: 77.13799999999999
    - type: mrr_at_100
      value: 77.488
    - type: mrr_at_1000
      value: 77.492
    - type: mrr_at_3
      value: 75.606
    - type: mrr_at_5
      value: 76.29599999999999
    - type: ndcg_at_1
      value: 69.02000000000001
    - type: ndcg_at_10
      value: 80.81099999999999
    - type: ndcg_at_100
      value: 82.298
    - type: ndcg_at_1000
      value: 82.403
    - type: ndcg_at_3
      value: 77.472
    - type: ndcg_at_5
      value: 78.892
    - type: precision_at_1
      value: 69.02000000000001
    - type: precision_at_10
      value: 9.336
    - type: precision_at_100
      value: 0.9990000000000001
    - type: precision_at_1000
      value: 0.101
    - type: precision_at_3
      value: 27.924
    - type: precision_at_5
      value: 17.492
    - type: recall_at_1
      value: 68.862
    - type: recall_at_10
      value: 92.308
    - type: recall_at_100
      value: 98.84100000000001
    - type: recall_at_1000
      value: 99.684
    - type: recall_at_3
      value: 83.193
    - type: recall_at_5
      value: 86.617
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/DuRetrieval
      name: MTEB DuRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 25.063999999999997
    - type: map_at_10
      value: 78.02
    - type: map_at_100
      value: 81.022
    - type: map_at_1000
      value: 81.06
    - type: map_at_3
      value: 53.613
    - type: map_at_5
      value: 68.008
    - type: mrr_at_1
      value: 87.8
    - type: mrr_at_10
      value: 91.827
    - type: mrr_at_100
      value: 91.913
    - type: mrr_at_1000
      value: 91.915
    - type: mrr_at_3
      value: 91.508
    - type: mrr_at_5
      value: 91.758
    - type: ndcg_at_1
      value: 87.8
    - type: ndcg_at_10
      value: 85.753
    - type: ndcg_at_100
      value: 88.82900000000001
    - type: ndcg_at_1000
      value: 89.208
    - type: ndcg_at_3
      value: 84.191
    - type: ndcg_at_5
      value: 83.433
    - type: precision_at_1
      value: 87.8
    - type: precision_at_10
      value: 41.33
    - type: precision_at_100
      value: 4.8
    - type: precision_at_1000
      value: 0.48900000000000005
    - type: precision_at_3
      value: 75.767
    - type: precision_at_5
      value: 64.25999999999999
    - type: recall_at_1
      value: 25.063999999999997
    - type: recall_at_10
      value: 87.357
    - type: recall_at_100
      value: 97.261
    - type: recall_at_1000
      value: 99.309
    - type: recall_at_3
      value: 56.259
    - type: recall_at_5
      value: 73.505
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/EcomRetrieval
      name: MTEB EcomRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 46.800000000000004
    - type: map_at_10
      value: 56.898
    - type: map_at_100
      value: 57.567
    - type: map_at_1000
      value: 57.593
    - type: map_at_3
      value: 54.167
    - type: map_at_5
      value: 55.822
    - type: mrr_at_1
      value: 46.800000000000004
    - type: mrr_at_10
      value: 56.898
    - type: mrr_at_100
      value: 57.567
    - type: mrr_at_1000
      value: 57.593
    - type: mrr_at_3
      value: 54.167
    - type: mrr_at_5
      value: 55.822
    - type: ndcg_at_1
      value: 46.800000000000004
    - type: ndcg_at_10
      value: 62.07
    - type: ndcg_at_100
      value: 65.049
    - type: ndcg_at_1000
      value: 65.666
    - type: ndcg_at_3
      value: 56.54
    - type: ndcg_at_5
      value: 59.492999999999995
    - type: precision_at_1
      value: 46.800000000000004
    - type: precision_at_10
      value: 7.84
    - type: precision_at_100
      value: 0.9169999999999999
    - type: precision_at_1000
      value: 0.096
    - type: precision_at_3
      value: 21.133
    - type: precision_at_5
      value: 14.099999999999998
    - type: recall_at_1
      value: 46.800000000000004
    - type: recall_at_10
      value: 78.4
    - type: recall_at_100
      value: 91.7
    - type: recall_at_1000
      value: 96.39999999999999
    - type: recall_at_3
      value: 63.4
    - type: recall_at_5
      value: 70.5
  - task:
      type: Classification
    dataset:
      type: C-MTEB/IFlyTek-classification
      name: MTEB IFlyTek
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 47.98768757214313
    - type: f1
      value: 35.23884426992269
  - task:
      type: Classification
    dataset:
      type: C-MTEB/JDReview-classification
      name: MTEB JDReview
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 86.97936210131333
    - type: ap
      value: 56.292679530375736
    - type: f1
      value: 81.87001614762136
  - task:
      type: STS
    dataset:
      type: C-MTEB/LCQMC
      name: MTEB LCQMC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 71.17149643620844
    - type: cos_sim_spearman
      value: 77.48040046337948
    - type: euclidean_pearson
      value: 76.32337539923347
    - type: euclidean_spearman
      value: 77.4804004621894
    - type: manhattan_pearson
      value: 76.33275226275444
    - type: manhattan_spearman
      value: 77.48979843086128
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/Mmarco-reranking
      name: MTEB MMarcoReranking
      config: default
      split: dev
      revision: None
    metrics:
    - type: map
      value: 27.966807589556826
    - type: mrr
      value: 26.92023809523809
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/MMarcoRetrieval
      name: MTEB MMarcoRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 66.15100000000001
    - type: map_at_10
      value: 75.048
    - type: map_at_100
      value: 75.374
    - type: map_at_1000
      value: 75.386
    - type: map_at_3
      value: 73.26700000000001
    - type: map_at_5
      value: 74.39
    - type: mrr_at_1
      value: 68.381
    - type: mrr_at_10
      value: 75.644
    - type: mrr_at_100
      value: 75.929
    - type: mrr_at_1000
      value: 75.93900000000001
    - type: mrr_at_3
      value: 74.1
    - type: mrr_at_5
      value: 75.053
    - type: ndcg_at_1
      value: 68.381
    - type: ndcg_at_10
      value: 78.669
    - type: ndcg_at_100
      value: 80.161
    - type: ndcg_at_1000
      value: 80.46799999999999
    - type: ndcg_at_3
      value: 75.3
    - type: ndcg_at_5
      value: 77.172
    - type: precision_at_1
      value: 68.381
    - type: precision_at_10
      value: 9.48
    - type: precision_at_100
      value: 1.023
    - type: precision_at_1000
      value: 0.105
    - type: precision_at_3
      value: 28.299999999999997
    - type: precision_at_5
      value: 17.98
    - type: recall_at_1
      value: 66.15100000000001
    - type: recall_at_10
      value: 89.238
    - type: recall_at_100
      value: 96.032
    - type: recall_at_1000
      value: 98.437
    - type: recall_at_3
      value: 80.318
    - type: recall_at_5
      value: 84.761
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_intent
      name: MTEB MassiveIntentClassification (zh-CN)
      config: zh-CN
      split: test
      revision: 31efe3c427b0bae9c22cbb560b8f15491cc6bed7
    metrics:
    - type: accuracy
      value: 68.26160053799597
    - type: f1
      value: 65.96949453305112
  - task:
      type: Classification
    dataset:
      type: mteb/amazon_massive_scenario
      name: MTEB MassiveScenarioClassification (zh-CN)
      config: zh-CN
      split: test
      revision: 7d571f92784cd94a019292a1f45445077d0ef634
    metrics:
    - type: accuracy
      value: 73.12037659717554
    - type: f1
      value: 72.69052407105445
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/MedicalRetrieval
      name: MTEB MedicalRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 50.1
    - type: map_at_10
      value: 56.489999999999995
    - type: map_at_100
      value: 57.007
    - type: map_at_1000
      value: 57.06400000000001
    - type: map_at_3
      value: 55.25
    - type: map_at_5
      value: 55.93
    - type: mrr_at_1
      value: 50.3
    - type: mrr_at_10
      value: 56.591
    - type: mrr_at_100
      value: 57.108000000000004
    - type: mrr_at_1000
      value: 57.165
    - type: mrr_at_3
      value: 55.35
    - type: mrr_at_5
      value: 56.03
    - type: ndcg_at_1
      value: 50.1
    - type: ndcg_at_10
      value: 59.419999999999995
    - type: ndcg_at_100
      value: 62.28900000000001
    - type: ndcg_at_1000
      value: 63.9
    - type: ndcg_at_3
      value: 56.813
    - type: ndcg_at_5
      value: 58.044
    - type: precision_at_1
      value: 50.1
    - type: precision_at_10
      value: 6.859999999999999
    - type: precision_at_100
      value: 0.828
    - type: precision_at_1000
      value: 0.096
    - type: precision_at_3
      value: 20.433
    - type: precision_at_5
      value: 12.86
    - type: recall_at_1
      value: 50.1
    - type: recall_at_10
      value: 68.60000000000001
    - type: recall_at_100
      value: 82.8
    - type: recall_at_1000
      value: 95.7
    - type: recall_at_3
      value: 61.3
    - type: recall_at_5
      value: 64.3
  - task:
      type: Classification
    dataset:
      type: C-MTEB/MultilingualSentiment-classification
      name: MTEB MultilingualSentiment
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 73.41000000000001
    - type: f1
      value: 72.87768282499509
  - task:
      type: PairClassification
    dataset:
      type: C-MTEB/OCNLI
      name: MTEB Ocnli
      config: default
      split: validation
      revision: None
    metrics:
    - type: cos_sim_accuracy
      value: 73.4163508391987
    - type: cos_sim_ap
      value: 78.51058998215277
    - type: cos_sim_f1
      value: 75.3875968992248
    - type: cos_sim_precision
      value: 69.65085049239033
    - type: cos_sim_recall
      value: 82.15417106652588
    - type: dot_accuracy
      value: 73.4163508391987
    - type: dot_ap
      value: 78.51058998215277
    - type: dot_f1
      value: 75.3875968992248
    - type: dot_precision
      value: 69.65085049239033
    - type: dot_recall
      value: 82.15417106652588
    - type: euclidean_accuracy
      value: 73.4163508391987
    - type: euclidean_ap
      value: 78.51058998215277
    - type: euclidean_f1
      value: 75.3875968992248
    - type: euclidean_precision
      value: 69.65085049239033
    - type: euclidean_recall
      value: 82.15417106652588
    - type: manhattan_accuracy
      value: 73.03735787763942
    - type: manhattan_ap
      value: 78.4190891700083
    - type: manhattan_f1
      value: 75.32592950265573
    - type: manhattan_precision
      value: 69.3950177935943
    - type: manhattan_recall
      value: 82.36536430834214
    - type: max_accuracy
      value: 73.4163508391987
    - type: max_ap
      value: 78.51058998215277
    - type: max_f1
      value: 75.3875968992248
  - task:
      type: Classification
    dataset:
      type: C-MTEB/OnlineShopping-classification
      name: MTEB OnlineShopping
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 91.81000000000002
    - type: ap
      value: 89.35809579688139
    - type: f1
      value: 91.79220350456818
  - task:
      type: STS
    dataset:
      type: C-MTEB/PAWSX
      name: MTEB PAWSX
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 30.10755999973859
    - type: cos_sim_spearman
      value: 36.221732138848864
    - type: euclidean_pearson
      value: 36.41120179336658
    - type: euclidean_spearman
      value: 36.221731188009436
    - type: manhattan_pearson
      value: 36.34865300346968
    - type: manhattan_spearman
      value: 36.17696483080459
  - task:
      type: STS
    dataset:
      type: C-MTEB/QBQTC
      name: MTEB QBQTC
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 36.778975708100226
    - type: cos_sim_spearman
      value: 38.733929926753724
    - type: euclidean_pearson
      value: 37.13383498228113
    - type: euclidean_spearman
      value: 38.73374886550868
    - type: manhattan_pearson
      value: 37.175732896552404
    - type: manhattan_spearman
      value: 38.74120541657908
  - task:
      type: STS
    dataset:
      type: mteb/sts22-crosslingual-sts
      name: MTEB STS22 (zh)
      config: zh
      split: test
      revision: 6d1ba47164174a496b7fa5d3569dae26a6813b80
    metrics:
    - type: cos_sim_pearson
      value: 65.97095922825076
    - type: cos_sim_spearman
      value: 68.87452938308421
    - type: euclidean_pearson
      value: 67.23101642424429
    - type: euclidean_spearman
      value: 68.87452938308421
    - type: manhattan_pearson
      value: 67.29909334410189
    - type: manhattan_spearman
      value: 68.89807985930508
  - task:
      type: STS
    dataset:
      type: C-MTEB/STSB
      name: MTEB STSB
      config: default
      split: test
      revision: None
    metrics:
    - type: cos_sim_pearson
      value: 78.98860630733722
    - type: cos_sim_spearman
      value: 79.36601601355665
    - type: euclidean_pearson
      value: 78.77295944956447
    - type: euclidean_spearman
      value: 79.36585127278974
    - type: manhattan_pearson
      value: 78.82060736131619
    - type: manhattan_spearman
      value: 79.4395526421926
  - task:
      type: Reranking
    dataset:
      type: C-MTEB/T2Reranking
      name: MTEB T2Reranking
      config: default
      split: dev
      revision: None
    metrics:
    - type: map
      value: 66.40501824507894
    - type: mrr
      value: 76.18463933756757
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/T2Retrieval
      name: MTEB T2Retrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 27.095000000000002
    - type: map_at_10
      value: 76.228
    - type: map_at_100
      value: 79.865
    - type: map_at_1000
      value: 79.935
    - type: map_at_3
      value: 53.491
    - type: map_at_5
      value: 65.815
    - type: mrr_at_1
      value: 89.554
    - type: mrr_at_10
      value: 92.037
    - type: mrr_at_100
      value: 92.133
    - type: mrr_at_1000
      value: 92.137
    - type: mrr_at_3
      value: 91.605
    - type: mrr_at_5
      value: 91.88
    - type: ndcg_at_1
      value: 89.554
    - type: ndcg_at_10
      value: 83.866
    - type: ndcg_at_100
      value: 87.566
    - type: ndcg_at_1000
      value: 88.249
    - type: ndcg_at_3
      value: 85.396
    - type: ndcg_at_5
      value: 83.919
    - type: precision_at_1
      value: 89.554
    - type: precision_at_10
      value: 41.792
    - type: precision_at_100
      value: 4.997
    - type: precision_at_1000
      value: 0.515
    - type: precision_at_3
      value: 74.795
    - type: precision_at_5
      value: 62.675000000000004
    - type: recall_at_1
      value: 27.095000000000002
    - type: recall_at_10
      value: 82.694
    - type: recall_at_100
      value: 94.808
    - type: recall_at_1000
      value: 98.30600000000001
    - type: recall_at_3
      value: 55.156000000000006
    - type: recall_at_5
      value: 69.19
  - task:
      type: Classification
    dataset:
      type: C-MTEB/TNews-classification
      name: MTEB TNews
      config: default
      split: validation
      revision: None
    metrics:
    - type: accuracy
      value: 51.929
    - type: f1
      value: 50.16876489927282
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/ThuNewsClusteringP2P
      name: MTEB ThuNewsClusteringP2P
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 61.404157724658894
  - task:
      type: Clustering
    dataset:
      type: C-MTEB/ThuNewsClusteringS2S
      name: MTEB ThuNewsClusteringS2S
      config: default
      split: test
      revision: None
    metrics:
    - type: v_measure
      value: 57.11418384351802
  - task:
      type: Retrieval
    dataset:
      type: C-MTEB/VideoRetrieval
      name: MTEB VideoRetrieval
      config: default
      split: dev
      revision: None
    metrics:
    - type: map_at_1
      value: 52.1
    - type: map_at_10
      value: 62.956999999999994
    - type: map_at_100
      value: 63.502
    - type: map_at_1000
      value: 63.51599999999999
    - type: map_at_3
      value: 60.75000000000001
    - type: map_at_5
      value: 62.195
    - type: mrr_at_1
      value: 52.0
    - type: mrr_at_10
      value: 62.907000000000004
    - type: mrr_at_100
      value: 63.452
    - type: mrr_at_1000
      value: 63.466
    - type: mrr_at_3
      value: 60.699999999999996
    - type: mrr_at_5
      value: 62.144999999999996
    - type: ndcg_at_1
      value: 52.1
    - type: ndcg_at_10
      value: 67.93299999999999
    - type: ndcg_at_100
      value: 70.541
    - type: ndcg_at_1000
      value: 70.91300000000001
    - type: ndcg_at_3
      value: 63.468
    - type: ndcg_at_5
      value: 66.08800000000001
    - type: precision_at_1
      value: 52.1
    - type: precision_at_10
      value: 8.34
    - type: precision_at_100
      value: 0.955
    - type: precision_at_1000
      value: 0.098
    - type: precision_at_3
      value: 23.767
    - type: precision_at_5
      value: 15.540000000000001
    - type: recall_at_1
      value: 52.1
    - type: recall_at_10
      value: 83.39999999999999
    - type: recall_at_100
      value: 95.5
    - type: recall_at_1000
      value: 98.4
    - type: recall_at_3
      value: 71.3
    - type: recall_at_5
      value: 77.7
  - task:
      type: Classification
    dataset:
      type: C-MTEB/waimai-classification
      name: MTEB Waimai
      config: default
      split: test
      revision: None
    metrics:
    - type: accuracy
      value: 87.12
    - type: ap
      value: 70.85284793227382
    - type: f1
      value: 85.55420883566512
---

## stella model

**新闻 | News**

**[2023-10-19]** 开源stella-base-en-v2 使用简单，**不需要任何前缀文本**。
Release stella-base-en-v2. This model **does not need any prefix text**.\
**[2023-10-12]** 开源stella-base-zh-v2和stella-large-zh-v2, 效果更好且使用简单，**不需要任何前缀文本**。
Release stella-base-zh-v2 and stella-large-zh-v2. The 2 models have better performance
and **do not need any prefix text**.\
**[2023-09-11]** 开源stella-base-zh和stella-large-zh

stella是一个通用的文本编码模型，主要有以下模型：

|     Model Name     | Model Size (GB) | Dimension | Sequence Length | Language | Need instruction for retrieval? |
|:------------------:|:---------------:|:---------:|:---------------:|:--------:|:-------------------------------:|
| stella-base-en-v2  |       0.2       |    768    |       512       | English  |               No                |
| stella-large-zh-v2 |      0.65       |   1024    |      1024       | Chinese  |               No                |
| stella-base-zh-v2  |       0.2       |    768    |      1024       | Chinese  |               No                |
|  stella-large-zh   |      0.65       |   1024    |      1024       | Chinese  |               Yes               |
|   stella-base-zh   |       0.2       |    768    |      1024       | Chinese  |               Yes               |

完整的训练思路和训练过程已记录在[博客1](https://zhuanlan.zhihu.com/p/655322183)和[博客2](https://zhuanlan.zhihu.com/p/662209559)，欢迎阅读讨论。

**训练数据：**

1. 开源数据(wudao_base_200GB[1]、m3e[2]和simclue[3])，着重挑选了长度大于512的文本
2. 在通用语料库上使用LLM构造一批(question, paragraph)和(sentence, paragraph)数据

**训练方法：**

1. 对比学习损失函数
2. 带有难负例的对比学习损失函数(分别基于bm25和vector构造了难负例)
3. EWC(Elastic Weights Consolidation)[4]
4. cosent loss[5]
5. 每一种类型的数据一个迭代器，分别计算loss进行更新

stella-v2在stella模型的基础上，使用了更多的训练数据，同时知识蒸馏等方法去除了前置的instruction(
比如piccolo的`查询:`, `结果:`, e5的`query:`和`passage:`)。

**初始权重：**\
stella-base-zh和stella-large-zh分别以piccolo-base-zh[6]和piccolo-large-zh作为基础模型，512-1024的position
embedding使用层次分解位置编码[7]进行初始化。\
感谢商汤科技研究院开源的[piccolo系列模型](https://huggingface.co/sensenova)。

stella is a general-purpose text encoder, which mainly includes the following models:

|     Model Name     | Model Size (GB) | Dimension | Sequence Length | Language | Need instruction for retrieval? |
|:------------------:|:---------------:|:---------:|:---------------:|:--------:|:-------------------------------:|
| stella-base-en-v2  |       0.2       |    768    |       512       | English  |               No                |
| stella-large-zh-v2 |      0.65       |   1024    |      1024       | Chinese  |               No                |
| stella-base-zh-v2  |       0.2       |    768    |      1024       | Chinese  |               No                |
|  stella-large-zh   |      0.65       |   1024    |      1024       | Chinese  |               Yes               |
|   stella-base-zh   |       0.2       |    768    |      1024       | Chinese  |               Yes               |

The training data mainly includes:

1. Open-source training data (wudao_base_200GB, m3e, and simclue), with a focus on selecting texts with lengths greater
   than 512.
2. A batch of (question, paragraph) and (sentence, paragraph) data constructed on a general corpus using LLM.

The loss functions mainly include:

1. Contrastive learning loss function
2. Contrastive learning loss function with hard negative examples (based on bm25 and vector hard negatives)
3. EWC (Elastic Weights Consolidation)
4. cosent loss

Model weight initialization:\
stella-base-zh and stella-large-zh use piccolo-base-zh and piccolo-large-zh as the base models, respectively, and the
512-1024 position embedding uses the initialization strategy of hierarchical decomposed position encoding.

Training strategy:\
One iterator for each type of data, separately calculating the loss.

Based on stella models, stella-v2 use more training data and remove instruction by Knowledge Distillation.

## Metric

#### C-MTEB leaderboard (Chinese)

|     Model Name     | Model Size (GB) | Dimension | Sequence Length | Average (35) | Classification (9) | Clustering (4) | Pair Classification (2) | Reranking (4) | Retrieval (8) | STS (8) |
|:------------------:|:---------------:|:---------:|:---------------:|:------------:|:------------------:|:--------------:|:-----------------------:|:-------------:|:-------------:|:-------:|
| stella-large-zh-v2 |      0.65       |   1024    |      1024       |    65.13     |       69.05        |     49.16      |          82.68          |     66.41     |     70.14     |  58.66  |
| stella-base-zh-v2  |       0.2       |    768    |      1024       |    64.36     |       68.29        |      49.4      |          79.95          |     66.1      |     70.08     |  56.92  |
|  stella-large-zh   |      0.65       |   1024    |      1024       |    64.54     |       67.62        |     48.65      |          78.72          |     65.98     |     71.02     |  58.3   |
|   stella-base-zh   |       0.2       |    768    |      1024       |    64.16     |       67.77        |      48.7      |          76.09          |     66.95     |     71.07     |  56.54  |

#### MTEB leaderboard (English)

|    Model Name     | Model Size (GB) | Dimension | Sequence Length | Average (56) | Classification (12) | Clustering (11) | Pair Classification (3) | Reranking (4) | Retrieval (15) | STS (10) | Summarization  (1) |
|:-----------------:|:---------------:|:---------:|:---------------:|:------------:|:-------------------:|:---------------:|:-----------------------:|:-------------:|:--------------:|:--------:|:------------------:|
| stella-base-en-v2 |       0.2       |    768    |       512       |    62.61     |        75.28        |      44.9       |          86.45          |     58.77     |      50.1      |  83.02   |       32.52        |

#### Reproduce our results

**C-MTEB:** 

```python
import torch
import numpy as np
from typing import List
from mteb import MTEB
from sentence_transformers import SentenceTransformer


class FastTextEncoder():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name).cuda().half().eval()
        self.model.max_seq_length = 512

    def encode(
            self,
            input_texts: List[str],
            *args,
            **kwargs
    ):
        new_sens = list(set(input_texts))
        new_sens.sort(key=lambda x: len(x), reverse=True)
        vecs = self.model.encode(
            new_sens, normalize_embeddings=True, convert_to_numpy=True, batch_size=256
        ).astype(np.float32)
        sen2arrid = {sen: idx for idx, sen in enumerate(new_sens)}
        vecs = vecs[[sen2arrid[sen] for sen in input_texts]]
        torch.cuda.empty_cache()
        return vecs


if __name__ == '__main__':
    model_name = "infgrad/stella-base-zh-v2"
    output_folder = "zh_mteb_results/stella-base-zh-v2"
    task_names = [t.description["name"] for t in MTEB(task_langs=['zh', 'zh-CN']).tasks]
    model = FastTextEncoder(model_name)
    for task in task_names:
        MTEB(tasks=[task], task_langs=['zh', 'zh-CN']).run(model, output_folder=output_folder)

```

**MTEB:**

You can use official script to reproduce our result. [scripts/run_mteb_english.py](https://github.com/embeddings-benchmark/mteb/blob/main/scripts/run_mteb_english.py)

#### Evaluation for long text

经过实际观察发现，C-MTEB的评测数据长度基本都是小于512的，
更致命的是那些长度大于512的文本，其重点都在前半部分
这里以CMRC2018的数据为例说明这个问题：

```
question: 《无双大蛇z》是谁旗下ω-force开发的动作游戏？

passage：《无双大蛇z》是光荣旗下ω-force开发的动作游戏，于2009年3月12日登陆索尼playstation3，并于2009年11月27日推......
```

passage长度为800多，大于512，但是对于这个question而言只需要前面40个字就足以检索，多的内容对于模型而言是一种噪声，反而降低了效果。\
简言之，现有数据集的2个问题：\
1）长度大于512的过少\
2）即便大于512，对于检索而言也只需要前512的文本内容\
导致**无法准确评估模型的长文本编码能力。**

为了解决这个问题，搜集了相关开源数据并使用规则进行过滤，最终整理了6份长文本测试集,他们分别是：

- CMRC2018，通用百科
- CAIL，法律阅读理解
- DRCD，繁体百科，已转简体
- Military，军工问答
- Squad，英文阅读理解，已转中文
- Multifieldqa_zh，清华的大模型长文本理解能力评测数据[9]

处理规则是选取答案在512长度之后的文本，短的测试数据会欠采样一下，长短文本占比约为1:2，所以模型既得理解短文本也得理解长文本。
除了Military数据集，我们提供了其他5个测试数据的下载地址：https://drive.google.com/file/d/1WC6EWaCbVgz-vPMDFH4TwAMkLyh5WNcN/view?usp=sharing

评测指标为Recall@5, 结果如下：

|     Dataset     | piccolo-base-zh | piccolo-large-zh | bge-base-zh | bge-large-zh | stella-base-zh | stella-large-zh | 
|:---------------:|:---------------:|:----------------:|:-----------:|:------------:|:--------------:|:---------------:|
|    CMRC2018     |      94.34      |      93.82       |    91.56    |    93.12     |     96.08      |      95.56      | 
|      CAIL       |      28.04      |      33.64       |    31.22    |    33.94     |     34.62      |      37.18      | 
|      DRCD       |      78.25      |       77.9       |    78.34    |    80.26     |     86.14      |      84.58      | 
|    Military     |      76.61      |      73.06       |    75.65    |    75.81     |     83.71      |      80.48      | 
|      Squad      |      91.21      |      86.61       |    87.87    |    90.38     |     93.31      |      91.21      | 
| Multifieldqa_zh |      81.41      |      83.92       |    83.92    |    83.42     |      79.9      |      80.4       | 
|   **Average**   |      74.98      |      74.83       |    74.76    |    76.15     |   **78.96**    |    **78.24**    | 

**注意：** 因为长文本评测数据数量稀少，所以构造时也使用了train部分，如果自行评测，请注意模型的训练数据以免数据泄露。

## Usage

#### stella 中文系列模型

stella-base-zh 和 stella-large-zh: 本模型是在piccolo基础上训练的，因此**用法和piccolo完全一致**
，即在检索重排任务上给query和passage加上`查询: `和`结果: `。对于短短匹配不需要做任何操作。

stella-base-zh-v2 和 stella-large-zh-v2: 本模型使用简单，**任何使用场景中都不需要加前缀文本**。

stella中文系列模型均使用mean pooling做为文本向量。

在sentence-transformer库中的使用方法：

```python
from sentence_transformers import SentenceTransformer

sentences = ["数据1", "数据2"]
model = SentenceTransformer('infgrad/stella-base-zh-v2')
print(model.max_seq_length)
embeddings_1 = model.encode(sentences, normalize_embeddings=True)
embeddings_2 = model.encode(sentences, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

直接使用transformers库：

```python
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

model = AutoModel.from_pretrained('infgrad/stella-base-zh-v2')
tokenizer = AutoTokenizer.from_pretrained('infgrad/stella-base-zh-v2')
sentences = ["数据1", "数据ABCDEFGH"]
batch_data = tokenizer(
    batch_text_or_text_pairs=sentences,
    padding="longest",
    return_tensors="pt",
    max_length=1024,
    truncation=True,
)
attention_mask = batch_data["attention_mask"]
model_output = model(**batch_data)
last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
vectors = normalize(vectors, norm="l2", axis=1, )
print(vectors.shape)  # 2,768
```

#### stella models for English

**Using Sentence-Transformers:**

```python
from sentence_transformers import SentenceTransformer

sentences = ["one car come", "one car go"]
model = SentenceTransformer('infgrad/stella-base-en-v2')
print(model.max_seq_length)
embeddings_1 = model.encode(sentences, normalize_embeddings=True)
embeddings_2 = model.encode(sentences, normalize_embeddings=True)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
```

**Using HuggingFace Transformers:**

```python
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

model = AutoModel.from_pretrained('infgrad/stella-base-en-v2')
tokenizer = AutoTokenizer.from_pretrained('infgrad/stella-base-en-v2')
sentences = ["one car come", "one car go"]
batch_data = tokenizer(
    batch_text_or_text_pairs=sentences,
    padding="longest",
    return_tensors="pt",
    max_length=512,
    truncation=True,
)
attention_mask = batch_data["attention_mask"]
model_output = model(**batch_data)
last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
vectors = normalize(vectors, norm="l2", axis=1, )
print(vectors.shape)  # 2,768
```

## Training Detail

**硬件：** 单卡A100-80GB

**环境：** torch1.13.*; transformers-trainer + deepspeed + gradient-checkpointing

**学习率：** 1e-6

**batch_size：** base模型为1024，额外增加20%的难负例；large模型为768，额外增加20%的难负例

**数据量：** 第一版模型约100万，其中用LLM构造的数据约有200K. LLM模型大小为13b。v2系列模型到了2000万训练数据。

## ToDoList

**评测的稳定性：**
评测过程中发现Clustering任务会和官方的结果不一致，大约有±0.0x的小差距，原因是聚类代码没有设置random_seed，差距可以忽略不计，不影响评测结论。

**更高质量的长文本训练和测试数据：** 训练数据多是用13b模型构造的，肯定会存在噪声。
测试数据基本都是从mrc数据整理来的，所以问题都是factoid类型，不符合真实分布。

**OOD的性能：** 虽然近期出现了很多向量编码模型，但是对于不是那么通用的domain，这一众模型包括stella、openai和cohere,
它们的效果均比不上BM25。

## Reference

1. https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab
2. https://github.com/wangyuxinwhy/uniem
3. https://github.com/CLUEbenchmark/SimCLUE
4. https://arxiv.org/abs/1612.00796
5. https://kexue.fm/archives/8847
6. https://huggingface.co/sensenova/piccolo-base-zh
7. https://kexue.fm/archives/7947
8. https://github.com/FlagOpen/FlagEmbedding
9. https://github.com/THUDM/LongBench



