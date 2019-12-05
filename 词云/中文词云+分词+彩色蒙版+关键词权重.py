# -*- coding: utf-8 -*-

from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba.analyse

# 打开文本
text = open('./text/xyj.txt',encoding='UTF-8').read()

# 提取关键词和权重
freq=jieba.analyse.extract_tags(text,topK=200,withWeight=True)
freq = {i[0]: i[1] for i in freq}

# 中文分词
text=' '.join(jieba.cut(text))

# 启用彩色蒙版
mask=np.array(Image.open('./mask/color_mask.png'))
wc = WordCloud(mask=mask,font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate_from_frequencies(freq)

# 从图片中生成颜色
image_colors=ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('./img/word7.png')