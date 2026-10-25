#!/usr/bin/env python3
"""Create original editable diagrams and an analytical capacity plot."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'articles/images/llm-inference'
BLUE = '#E7EFF7'
GRAY = '#F3F5F6'


class Diagram:
    def __init__(self, title, width=1240, height=640):
        self.tree = ET.Element('mxfile', host='app.diagrams.net')
        page = ET.SubElement(self.tree, 'diagram', name=title)
        model = ET.SubElement(page, 'mxGraphModel', page='1', pageWidth=str(width),
                              pageHeight=str(height), background='#ffffff')
        self.root = ET.SubElement(model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        ET.SubElement(self.root, 'mxCell', id='1', parent='0')
        self.index = 1
        self.box(title, 20, 10, width - 40, 50, fill='none',
                 extra='strokeColor=none;fontSize=29;fontStyle=1;')

    def box(self, text, x, y, width, height, fill=BLUE, extra=''):
        self.index += 1
        cell = ET.SubElement(self.root, 'mxCell', id=str(self.index), value=text,
            style=f'rounded=0;whiteSpace=wrap;html=1;fontFamily=Arial;fontSize=23;'
                  f'fontColor=#253542;strokeColor=#708798;fillColor={fill};spacing=10;{extra}',
            vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(width),
                      height=str(height), **{'as': 'geometry'})
        return str(self.index)

    def text(self, label, x, y, width, height, size=21):
        return self.box(label, x, y, width, height, 'none',
                        f'strokeColor=none;fontSize={size};')

    def edge(self, source, target, text='', extra='', points=()):
        self.index += 1
        cell = ET.SubElement(self.root, 'mxCell', id=str(self.index), value=text,
            source=source, target=target, edge='1', parent='1',
            style='edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;'
                  'html=1;strokeColor=#486579;strokeWidth=2;endArrow=block;'
                  'fontFamily=Arial;fontSize=19;labelBackgroundColor=#ffffff;'
                  f'exitX=1;exitY=0.5;entryX=0;entryY=0.5;{extra}')
        geometry = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
        if points:
            array = ET.SubElement(geometry, 'Array', **{'as': 'points'})
            for x, y in points:
                ET.SubElement(array, 'mxPoint', x=str(x), y=str(y))

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        ET.indent(self.tree)
        ET.ElementTree(self.tree).write(OUT / f'{name}.drawio', encoding='utf-8', xml_declaration=True)


def diagrams():
    d = Diagram('一段输入如何变成下一个 token', 1400, 600)
    labels = ('文本<br>Tokenizer', 'Token IDs', 'Embedding',
              'L 个<br>Decoder Block', 'Final RMSNorm<br>LM Head / logits', '采样<br>新 token')
    boxes = [d.box(label, 20 + 230 * index, 140, 200, 110)
             for index, label in enumerate(labels)]
    for source, target in zip(boxes, boxes[1:]):
        d.edge(source, target)
    sequence = d.box('追加到输出序列<br>下一步只送新增 token', 930, 380, 400, 100, GRAY)
    d.edge(boxes[-1], sequence, extra='exitX=0.5;exitY=1;entryX=0.85;entryY=0;')
    d.edge(sequence, boxes[2], 'Decode',
           extra='exitX=0;exitY=0.5;entryX=0.5;entryY=1;', points=((580, 430),))
    d.text('Prefill 处理已有 prompt；Decode 的历史上下文由各层 KV Cache 提供。',
           30, 520, 1340, 50)
    d.save('01-generation-path')

    d = Diagram('一个 Pre-Norm Decoder Block', 1240, 660)
    x = d.box('输入 X<br>[B, S, H]', 20, 130, 190, 80, GRAY)
    norm1 = d.box('RMSNorm', 280, 130, 210, 80, GRAY)
    attn = d.box('Attention<br>MHA + RoPE', 580, 120, 250, 100)
    add1 = d.box('+', 950, 140, 60, 60, GRAY, 'ellipse;fontSize=34;')
    norm2 = d.box('RMSNorm', 850, 360, 250, 80, GRAY)
    ffn = d.box('FFN<br>SwiGLU', 490, 350, 270, 100)
    add2 = d.box('+', 290, 370, 60, 60, GRAY, 'ellipse;fontSize=34;')
    y = d.box('输出 Y<br>[B, S, H]', 20, 360, 190, 80, GRAY)
    for source, target in ((x, norm1), (norm1, attn), (attn, add1)):
        d.edge(source, target)
    d.edge(x, add1, '残差', extra='exitX=0.5;exitY=0;entryX=0.5;entryY=0;',
           points=((110, 90), (980, 90)))
    d.edge(add1, norm2, extra='exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
    d.edge(norm2, ffn, extra='exitX=0;exitY=0.5;entryX=1;entryY=0.5;')
    d.edge(ffn, add2, extra='exitX=0;exitY=0.5;entryX=1;entryY=0.5;')
    d.edge(add2, y, extra='exitX=0;exitY=0.5;entryX=1;entryY=0.5;')
    d.edge(add1, add2, '残差', extra='exitX=1;exitY=0.5;entryX=0.5;entryY=1;',
           points=((1180, 170), (1180, 520), (320, 520)))
    d.text('Attention：Q/K/V 投影、Q/K 旋转、因果注意力和输出投影', 30, 570, 1180, 40)
    d.text('SwiGLU：gate / up 两组投影，激活与逐元素乘，再做 down 投影', 30, 610, 1180, 40)
    d.save('02-decoder-block')

    d = Diagram('Decode 只算新增位置 但仍需访问历史状态', 1240, 650)
    token = d.box('新增隐藏向量<br>[B, 1, H]', 20, 170, 200, 100, GRAY)
    projection = d.box('Q / K / V<br>线性投影', 300, 170, 220, 100)
    rope = d.box('Q / K 做 RoPE<br>V 不旋转', 610, 170, 260, 100)
    attention = d.box('Attention<br>得到当前输出', 980, 170, 220, 100)
    cache = d.box('本层 KV Cache<br>历史状态 + 新 K/V<br>可见长度 P+1', 590, 420, 300, 130, GRAY)
    d.edge(token, projection)
    d.edge(projection, rope)
    d.edge(rope, attention, 'Q', extra='exitX=1;exitY=0.3;entryX=0;entryY=0.3;')
    d.edge(rope, cache, '追加新 K/V', extra='exitX=0.5;exitY=1;entryX=0.5;entryY=0;')
    d.edge(cache, attention, '读取可见 K/V',
           extra='exitX=1;exitY=0.5;entryX=0.5;entryY=1;', points=((1090, 480),))
    d.text('P 是已有缓存长度；每层独立保存 K/V；历史 K 不重复旋转。', 20, 580, 1200, 50)
    d.save('03-kv-cache')

    d = Diagram('同一组投影权重面对的输入形状不同', 1240, 690)
    for mode, y, rows in (('Prefill', 100, 'B×S'), ('Decode', 380, 'B')):
        height = 170 if mode == 'Prefill' else 70
        top = y if mode == 'Prefill' else y + 50
        d.text(mode, 10, y + 50, 170, 60, 26)
        d.box(f'输入 X<br>[{rows}, H]', 200, top, 280, height, GRAY)
        d.text('×', 500, y + 50, 50, 60, 32)
        d.box('权重 W<br>[H, H]', 590, y, 180, 170)
        d.text('=', 800, y + 50, 50, 60, 32)
        d.box(f'输出 Y<br>[{rows}, H]', 890, top, 280, height, GRAY)
    d.text('示意的是线性投影；B 很小时，Decode 的矩阵行数很少。', 20, 590, 1200, 40)
    d.text('Attention 中：Prefill 的 query 长度为 S，单步 Decode 的 query 长度为 1。',
           20, 630, 1200, 40)
    d.save('04-prefill-decode')


def capacity_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = json.loads((ROOT / 'data/llm-inference/estimates.json').read_text())
    fig, axis = plt.subplots(figsize=(8.8, 4.5), constrained_layout=True)
    for heads, label, color in ((32, 'MHA: 32 KV heads', '#476DA7'),
                                 (8, 'GQA: 8 KV heads', '#237F73')):
        selected = [row for row in report['cache_capacity_examples'] if row['kv_heads'] == heads]
        x = [row['tokens'] / 1024 for row in selected]
        y = [row['cache_bytes'] / 2**30 for row in selected]
        axis.plot(x, y, marker='o', linewidth=2, color=color, label=label)
        for a, b in zip(x, y):
            axis.annotate(f'{b:g} GiB', (a, b), xytext=(0, 9), textcoords='offset points', ha='center')
    axis.set_title('KV cache capacity: analytical estimate\n32 layers | 32 query heads | d=128 | batch=1 | 2 bytes/element', fontsize=11)
    axis.set_xlabel('Cached context length (K tokens; K = 1024)')
    axis.set_ylabel('KV cache capacity (GiB)')
    axis.set_xticks((8, 16, 32))
    axis.set_ylim(0, 18)
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=0.2)
    axis.legend(loc='upper left')
    fig.savefig(OUT / '05-kv-capacity.png', dpi=190)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='requires matplotlib')
    args = parser.parse_args()
    diagrams()
    if args.plot:
        capacity_plot()
