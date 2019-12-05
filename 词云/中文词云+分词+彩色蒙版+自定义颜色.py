# -*- coding: utf-8 -*-

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import jieba

# 打开文本
text = open('./text/xyj.txt',encoding='UTF-8').read()

# 中文分词
text=' '.join(jieba.cut(text))

# 自定义颜色函数
def random_color(word, font_size, position, orientation, font_path, random_state):
	s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
	return s

# 启用彩色蒙版
mask=np.array(Image.open('./mask/color_mask.png'))
wc = WordCloud(color_func=random_color,mask=mask,font_path='Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)

# 显示词云
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 保存到文件
wc.to_file('./img/word6.png')