NVIDIA AI-AGENT夏季训练营
=================
# DjangoNIMBot
通过调用nvidia nim平台的大模型api和django web框架实现的一个简单的ai对话应用

* 项目名称：AI-AGENT夏季训练营 — RAG智能对话机器人
* 报告日期：2024年8月18日
* 项目负责人：金日耀

项目概述
-------
使用NVIDIA NIM平台的大模型API和django web框架实现了一个简单的网页问答机器人、支持基于RAG的生成式模型，同时也使用了LLAMA3和sdxl-turbo两个模型的API接口实现了简单的对话和图像生成功能。

技术方案与实施步骤
-------
* 模型选择：选择的三个模型分别是ai-phi-3-small-128k-instruct、llama-3.1-405b-instruc以及sdxl-turbo、RAG模型是参照day1例子中的模型、尝试使用llama-3.1-405b-instruc没成功、选择sdxl-turbo的理由是在Try NVIDIA NIM APIs有比较详细的使用例子，其他图像生成模型的接口文档太长了不适合短时间掌握。
* 数据的构建：数据整理了训练营的简介信息和每天的日程安排、用chatapt做了数据清洗(原有数据的各种符号以及并列类似排比句的信息直接作为RAG数据模型生成的回答很奇怪，用GPT总结了一下再输入给模型才正常)。
* 我通过交互式UI调用不同的接口实现不同功能的整个（关键因素是缺少能通过用户的描述判断其意图的模型或者是什么东东，抑或是phi3这个多模态模型有的输出中能有什么标识符来判断结果是图片还是纯文本抑或是文件以便于之后UI程序的展示处理）

实施步骤
-------
整个项目代码在仓库中、我抽取了必要的依赖信息写在了requirements.txt文件里（可能有些落了依据报错信息也能很好的安装上）

linux系统在项目的DjangoNIMBot/django_nimbot目录下直接
```
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```
会弹出一个输入框，在数据框中输入NIM平台申请的key

然后访问http://127.0.0.1:8000/api/chat_ui/

RAG实现的关键代码就在day1课件的demo里，将它集成在整个项目代码中，因为向量信息数据库只需要生成一次可以把这块代码封装在django工程的app load步骤中，这样只有启动工程的时候才会执行该步骤，提高效率。部署在了[部署地址](http://175.24.132.143:8000/api/chat_ui/)这个地址（不要开网页代理，公网IP之后我会更换掉）

项目成果与展示
-------
将业务数据统计作为向量数据库，这个小项目可以做一个简单的客服机器人
![image](https://github.com/user-attachments/assets/34505b62-c788-48d9-a922-775d7e45f71b)
![image](https://github.com/user-attachments/assets/c9e9d8d4-d384-451e-b7e7-c2c031f8609b)
![image](https://github.com/user-attachments/assets/710c056f-100d-4acd-a810-9a6dbe61c756)

问题与解决方案
-------
问题分析：生成的向量数据信息对于数据内容和格式的依赖太高了、导致输出的内容很抽象
解决措施：用gpt总结了洗了一下数据

项目总结与展望
-------
* 整个项目过程中通过NIM平台提供的demo实现了一些简单的功能我个人还是挺感兴趣的，希望之后NIM平台的功能和使用文档能更丰富（吐槽有些模型的文档太难看了）
* 虽然使用不同的接口实现不同的相应但是没有实现一个真正的多模态
* 希望之后有机会能找到检测用户意图的方法或者更深入的去了解一下langchain_nvidia_ai_endpoints的接口
* RAG模型输出的数据太死板了、基本和源数据一样，希望之后有机会更深层次去探索RAG技术

附件与参考资料
-------
项目部署的网页地址可以访问尝试功能（不要开代理访问！之后会把ip换掉,这个机器没域名直接暴露公网ip太抽象了）
* http://175.24.132.143:8000/api/chat_ui/
* [NIM平台地址](https://build.nvidia.com/explore/discover)强烈推荐




