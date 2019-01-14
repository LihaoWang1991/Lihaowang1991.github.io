Like Google Cloud in Google and AWS in Amazon, Azure is the cloud computing platform provided by Microsoft. Its main services include computing, mobile services, storage services, data management, media services, machine learning, IoT, etc. 


[Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml) provides a cloud-based environment you can use to develop, train, test, deploy, manage, and track machine learning models. The process of a typical Azure machine learning is as below:

![](https://github.com/LihaoWang1991/lihaowang1991.github.io/blob/master/img/post-azure2.jpg)

Like most machine learning platforms, the supported language on Azure Machine Learning service is Python. It fully supports open-source technologies. That means you can use open-source Python packages such as TensorFlow and scikit-learn. If you are familiar with coding using Jupyter Notebook, then Azure Machine Learning service can be a good choice to you because it has the same programming interface which is called [Azure Notebooks](https://notebooks.azure.com/). Nevertheless, you can also code on your local Python IDE but you need to install [Azure Python SDK](https://docs.microsoft.com/zh-cn/python/api/overview/azure/ml/intro?view=azure-ml-py) packages at first. 

After learning the [official tutorials](https://docs.microsoft.com/en-us/azure/machine-learning/service/), I have deployed a little project I have done before on Azure Machine Leraning service. Here are some important notes I find during this project.

## Creating an Azure Machine Learning Workspace

The project begins by creating an Azure Machine Learning Workspace. 

The workspace is the top-level resource for Azure Machine Learning service. It provides a centralized place to work with all the artifacts you create when you use Azure Machine Learning service. The workspace keeps a list of compute targets that you can use to train your model. It also keeps a history of the training runs, including logs, metrics, output, and a snapshot of your scripts. You use this information to determine which training run produces the best model. 

The workspace is created on [Azure Portal](portal.azure.com) as below:

![](https://github.com/LihaoWang1991/lihaowang1991.github.io/blob/master/img/post-azure1.jpg)

Here is a brief explanation:

Resource group: A resource group is a container that holds related resources for an Azure solution.


## 正文

上次看了一篇 [《从一道网易面试题浅谈OC线程安全》](https://www.jianshu.com/p/cec2a41aa0e7) 的博客，主要内容是：

作者去网易面试，面试官出了一道面试题：下面代码会发生什么问题？

```objc
@property (nonatomic, strong) NSString *target;
//....
dispatch_queue_t queue = dispatch_queue_create("parallel", DISPATCH_QUEUE_CONCURRENT);
for (int i = 0; i < 1000000 ; i++) {
    dispatch_async(queue, ^{
        self.target = [NSString stringWithFormat:@"ksddkjalkjd%d",i];
    });
}
```

答案是：会 crash。

我们来看看对`target`属性（`strong`修饰）进行赋值，相当与 MRC 中的

```
- (void)setTarget:(NSString *)target {
    if (target == _target) return;
    id pre = _target;
    [target retain];//1.先保留新值
    _target = target;//2.再进行赋值
    [pre release];//3.释放旧值
}
```

因为在 *并行队列* `DISPATCH_QUEUE_CONCURRENT` 中*异步* `dispatch_async` 对 `target`属性进行赋值，就会导致 target 已经被 `release`了，还会执行 `release`。这就是向已释放内存对象发送消息而发生 crash 。


### 但是

我敲了这段代码，执行的时候发现并不会 crash~

```objc
@property (nonatomic, strong) NSString *target;
dispatch_queue_t queue = dispatch_queue_create("parallel", DISPATCH_QUEUE_CONCURRENT);
for (int i = 0; i < 1000000 ; i++) {
    dispatch_async(queue, ^{
    	// ‘ksddkjalkjd’删除了
        self.target = [NSString stringWithFormat:@"%d",i];
    });
}
```

原因就出在对 `self.target` 赋值的字符串上。博客的最后也提到了 - *‘上述代码的字符串改短一些，就不会崩溃’*，还有 `Tagged Pointer` 这个东西。

我们将上面的代码修改下：


```objc
NSString *str = [NSString stringWithFormat:@"%d", i];
NSLog(@"%d, %s, %p", i, object_getClassName(str), str);
self.target = str;
```

输出：

```
0, NSTaggedPointerString, 0x3015
```

发现这个字符串类型是 `NSTaggedPointerString`，那我们来看看 Tagged Pointer 是什么？

### Tagged Pointer

Tagged Pointer 详细的内容可以看这里 [深入理解Tagged Pointer](http://www.infoq.com/cn/articles/deep-understanding-of-tagged-pointer)。

Tagged Pointer 是一个能够提升性能、节省内存的有趣的技术。

- Tagged Pointer 专门用来存储小的对象，例如 **NSNumber** 和 **NSDate**（后来可以存储小字符串）
- Tagged Pointer 指针的值不再是地址了，而是真正的值。所以，实际上它不再是一个对象了，它只是一个披着对象皮的普通变量而已。
- 它的内存并不存储在堆中，也不需要 malloc 和 free，所以拥有极快的读取和创建速度。




### 参考：

- [What is Azure Machine Learning service?](https://docs.microsoft.com/en-us/azure/machine-learning/service/overview-what-is-azure-ml)

- [深入理解Tagged Pointer](http://www.infoq.com/cn/articles/deep-understanding-of-tagged-pointer)

- [【译】采用Tagged Pointer的字符串](http://www.cocoachina.com/ios/20150918/13449.html)
