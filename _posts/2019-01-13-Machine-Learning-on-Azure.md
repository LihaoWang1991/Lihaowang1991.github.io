---
layout:     post
title:      Machine learning on Azure
subtitle:   
date:       2019-01-13
author:     Lihao Wang
header-img: img/post-bg-universe.jpg
catalog: true
tags:
    - Cloud Computing
---


Like Google Cloud in Google and AWS in Amazon, Azure is the cloud computing platform provided by Microsoft. Its main services include computing, mobile services, storage services, data management, media services, machine learning, IoT, etc. 


[Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml) provides a cloud-based environment you can use to develop, train, test, deploy, manage, and track machine learning models. The process of a typical Azure machine learning is as below:

![](https://i.postimg.cc/wjsrnbjk/post-azure2.png)

Like most machine learning platforms, the supported language on Azure Machine Learning service is Python. It fully supports open-source technologies. That means you can use open-source Python packages such as TensorFlow and scikit-learn. If you are familiar with coding using Jupyter Notebook, then Azure Machine Learning service can be a good choice to you because it has the same programming interface which is called [Azure Notebooks](https://notebooks.azure.com/). Nevertheless, you can also code on your local Python IDE but you need to install [Azure Python SDK](https://docs.microsoft.com/zh-cn/python/api/overview/azure/ml/intro?view=azure-ml-py) packages at first. 

After learning the [official tutorials](https://docs.microsoft.com/en-us/azure/machine-learning/service/), I have deployed a little project I have done before on Azure Machine Leraning service. Here are some important notes I find during this project.

### Creating an Azure Machine Learning Workspace

The project begins by creating an Azure Machine Learning Workspace. 

The workspace is the top-level resource for Azure Machine Learning service. It provides a centralized place to work with all the artifacts you create when you use Azure Machine Learning service. The workspace keeps a list of compute targets that you can use to train your model. It also keeps a history of the training runs, including logs, metrics, output, and a snapshot of your scripts. You use this information to determine which training run produces the best model. 

The workspace is created on [Azure Portal](portal.azure.com) as below:

![](https://i.postimg.cc/wxtDfhnH/post-azure1.png)

**Resource group** means the container that holds related resources for an Azure solution.The machine learning workspace must be allocated to one resource group.


