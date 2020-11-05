## Post-Comparison Mitigation of Demographic Bias in Face Recognition Using Fair Score Normalization

<img src="Visualization2.png" height="400" align="right" >

Pattern Recognition Letters 2020

* [Research Paper (arXiv)](https://arxiv.org/abs/2002.03592)
* [Implementation](...)


## Table of Contents 

- [Abstract](#abstract)
- [Main Idea](#main-idea)
- [Results](#results)
- [Installation](#installation)
- [Citing](#citing)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## Abstract

Current face recognition systems achieve high progress on several benchmark tests. 
Despite this progress, recent works showed that these systems are strongly biased against demographic sub-groups.
Consequently, an easily integrable solution is needed to reduce the discriminatory effect of these biased systems. 
Previous work mainly focused on learning less biased face representations, which comes at the cost of a strongly degraded overall recognition performance. 
In this work, **we propose a novel unsupervised fair score normalization approach that is specifically designed to reduce the effect of bias in face recognition and subsequently lead to a significant overall performance boost**. 
Our hypothesis is **built on the notation of individual fairness by designing a normalization approach that leads to treating ”similar” individuals ”similarly”**. 
Experiments were conducted on three publicly available datasets captured under controlled and in-the-wild circumstances. 
Results demonstrate that our solution reduces demographic biases, e.g. by up to 82.7% in the case when gender is considered. 
Moreover, it mitigates the bias more consistently than existing works. 
In contrast to previous works, our fair normalization approach enhances the overall performance by up to 53.2% at false match rate 
of ![\Large 10^{-3}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-3}) and
up to 82.9% at a false match rate of ![\Large 10^{-5}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-5}). 
Additionally, it is easily integrable into existing recognition systems and not limited to face biometrics.

## Main idea

Explain main idea here

We first visually demonstrate the need for a more in-dividual treatment in face recognition systems.  More-over, we show that our approach is able to treat similarindividuals more similarly

<img src="Visualization.png" width="600" >

## Results

Experiments were conducted on three publicly available datasets (Adience, ColorFeret, and Morph).
Some results of this work are shown below (based on FaceNet embeddings).
For complete results, please take a look at the paper.

First, an analysis of the bias reduction of the proposed approach (Ours) in comparison with two previous works (SLF [28] and FTC [33]) is presented. 
The bias is measured in terms of STD of the class-wise FNMRs at a FMR of ![\Large 10^{-3}](https://latex.codecogs.com/gif.latex?\inline&space;10^{-3}). 
Unlike both related works, our proposed approach mitigates bias effectively and consistently.

<img src="BiasReductionFaceNet.png" width="600" >  

Moreover, the improvement of the overall face recognition performance is analysed and shown in the Figure below.
The verification performance is measured in terms of FNMR at several FMRs.
Even while making the recognition process more fair, in contrast to previous work, our approach consistently improves the global recognition performance.

<img src="ImprovementRecognitionPerformance.png" width="600" >

## Installation

TODO Jan


## Citing

If you use this code, please cite the following paper.


```
@article{DBLP:journals/corr/abs-2002-03592,
  author    = {Philipp Terh{\"{o}}rst and
               Jan Niklas Kolf and
               Naser Damer and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {Post-Comparison Mitigation of Demographic Bias in Face Recognition
               Using Fair Score Normalization},
  journal   = {CoRR},
  volume    = {abs/2002.03592},
  year      = {2020},
  url       = {https://arxiv.org/abs/2002.03592},
  archivePrefix = {arXiv},
  eprint    = {2002.03592},
  timestamp = {Wed, 12 Feb 2020 16:38:55 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2002-03592.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## Acknowledgement

This research work has been funded by the German Federal Ministery of Education and Research and the Hessen State
Ministry for Higher Education, Research and the Arts within their joint support of the National Research Center for Applied
Cybersecurity. 
Portions of the research in this paper use the FERET database of facial images collected under the FERET
program, sponsored by the Counterdrug Technology Development Program Office. 

## License 

This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
