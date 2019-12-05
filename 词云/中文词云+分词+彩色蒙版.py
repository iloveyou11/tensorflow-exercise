# -*- coding: utf-8 -*-

from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import jieba

# 打开文本
text = open('./text/xyj.txt',encoding='UTF-8').read()

# 中文分词
text=' '.join(jieba.cut(text))

# 启用彩色蒙版
mask=np.array(Image.open('./mask/color_mask.png'))
wc = WordCloud(mask=mask,font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

# 从图片中生成颜色
image_colors=ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('./img/word5.png')