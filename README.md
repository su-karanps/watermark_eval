# New Evaluation Metrics Capture Quality Degradation due to LLM Watermarking

[Karanpartap Singh](https://karanps.com), [James Zou](https://www.james-zou.com/)

**Stanford University, School of Engineering**

<hr>

## Abstract

With the increasing use of large-language models (LLMs) like ChatGPT, watermarking has emerged as a promising approach for tracing machine-generated content. However, research on LLM watermarking often relies on simple perplexity or diversity-based measures to assess the quality of watermarked text, which can mask important limitations in watermarking. Here we introduce two new easy-to-use methods for evaluating watermarking algorithms for LLMs: 1) evaluation by LLM-judger with specific guidelines; and 2) binary classification on text embeddings to distinguish between watermarked and unwatermarked text. We apply these methods to characterize the effectiveness of  current watermarking techniques. Our experiments, conducted across various datasets, reveal that current watermarking methods are moderately detectable by even simple classifiers, challenging the notion of watermarking subtlety. We also found, through the LLM judger, that watermarking impacts text quality, especially in degrading the coherence and depth of the response. Our findings underscore the trade-off between watermark robustness and text quality and highlight the importance of having more informative metrics to assess watermarking quality. 
Preprint: [arXiv](https://arxiv.org/abs/2312.02382)

## Getting Started

This repository contains notebooks detailing the two primary evaluation metrics: GPT-judger and binary classifiers, that were employed in our paper to assess the quality degradation and other impacts of LLM watermarks. 

## Citation

If you found our work helpful for your own research or applications, please cite it using the following BibTeX:
```bibtex
    @article{WatermarkEvaluation,
        title={New Evaluation Metrics Capture Quality Degradation due to LLM Watermarking},
        author={Karanpartap Singh, James Zou},
        journal={arXiv:2312.02382},
        year={2024}
    }
```
