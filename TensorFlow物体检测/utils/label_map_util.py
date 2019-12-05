# -*- coding: utf-8 -*-

import json

def create_category_index(categories):
  category_index = {}
  for cat in categories:
    category_index[cat['id']] = cat
  return category_index

def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
  categories = []
  for item in label_map:
    if use_display_name and 'display_name' in item:
      name = item['display_name']
    else:
      name = item['name']
    categories.append({'id': item['id'], 'name': name})
  return categories

def load_labelmap(path):
  with open(path, 'r') as fr:
    lines = fr.readlines()
    result = []

    cache = ''
    for line in lines:
      line = line.strip().strip('\n')
      if line.find('item') == 0:
        cache += '{'
      elif line.find('}') == 0:
        cache = cache[:-1] + '}'
        result.append(json.loads(cache))
        cache = ''
      else:
        line = line.split(':')
        line[0] = '"' + line[0] + '"'
        line = ':'.join(line)
        cache += line + ','

  return result