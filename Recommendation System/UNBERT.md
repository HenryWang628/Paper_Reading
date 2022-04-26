#### UNBERT:

- ##### Motivation: 

  现有模型只关注 in domain 数据，并从中提取出文本表征，限制了在冷启动场景的能力。UNBERT 模型通过引入预训练语言模型，使用丰富的语言知识提高文本表征能力，获取多粒度（词级别和新闻级别）的用户-新闻匹配信息。

- ##### Architecture:

  - 输入和输出：

     输入为给定用户的历史点击的新闻 ，以及候选新闻。 新闻是通过标题的词语表示。

      ![architecture](https://d3i71xaburhd42.cloudfront.net/a91553fe20e832c38e8f9ef4a4feb3d20eae0b0f/2-Figure2-1.png)

    

- ##### Dataset:

  - MIND dataset

- ##### Evaluation Metrics:

  ![UNBERT](https://d3i71xaburhd42.cloudfront.net/a91553fe20e832c38e8f9ef4a4feb3d20eae0b0f/3-Figure3-1.png)
