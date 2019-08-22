import fileinput
import os
from hanziconv import HanziConv
from gensim.models import word2vec
import re
import jieba
import time
from tqdm import tqdm
import mmap
from wordcloud import WordCloud
import numpy as np
import matplotlib.pyplot as plt


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# input = fileinput.FileInput(input_file, openhook=fileinput.hook_encoded("utf-8"))

# 执行命令行，提取语料到文件夹 wiki_corpus
trans_ = 1  # 转换一次就好啦，下次就不执行了
if not trans_:
    os.system(
        'python ./wikiextractor-master/WikiExtractor.py ' 
        '-o ./wiki_corpus '
        '-b 500M '
        '-q '
        './wiki_chinese20190720/zhwiki-20190720-pages-articles-multistream.xml'
    )


# 有很多其它符号，并且简体繁体混杂，所以只选出中英文和数字，并且中文转换成简体。
def data_clean(file_path, out_path):
    print('提取数字汉字英文', HanziConv.toSimplified('再进行繁簡轉換'))
    file_list = os.listdir(file_path)
    pattern = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9]+")
    with open(out_path, 'w', encoding='utf-8') as fw:
        for file in file_list:  # 多个文件融合到一起
            with open(file_path+'/'+file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    res = re.findall(pattern, line)
                    res = " ".join(res)
                    result = HanziConv.toSimplified(res)
                    # result = jieba.cut(result)
                    # symbol.update(res)
                    fw.write(result + '\n')


# jieba 分词，并将词语用空格隔开。
def data_cut(file_path, out_path):
    pre_word = ' '
    num_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    with open(out_path, 'w', encoding='utf-8') as fw:
        with open(file_path, 'r', encoding='utf-8') as fr:
            for line in tqdm(fr, total=num_lines):
                if not line.strip():
                    continue
                cut_list = jieba.cut(line)
                for word in cut_list:
                    if word == ' ' or word =='\n' or word == pre_word:
                        continue
                    fw.write(word + ' ')
                    pre_word = word
                fw.write('\n')  # 换行呀换行，不然后面就内存溢出啦，不换行 word2vec包使用流式处理也不好使
    return out_path


def vector_train(file_path, out_path):
    sentences = word2vec.LineSentence(file_path)
    num_lines = sum(1 for _ in sentences)  # 7433669
    print('----Training Word2Vec model with file', file_path)
    # batch_size = 5000
    # N = len(sentences)/batch_size
    model = word2vec.Word2Vec(tqdm(sentences, total=num_lines), min_count=5, size=300)
    print('----Saving Word2Vec model to file...', out_path)
    model.save(out_path)
    return out_path


def word2vec_model(model_path):
    model = word2vec.Word2Vec.load(model_path)
    sim_words = model.wv.most_similar('说', topn=10)
    for word, vec in sim_words:
        print('word:', word, 'vector', vec, 'similarity:', model.wv.similarity('说', word))


# data_clean('./wiki_corpus/AA', 'wiki_corpus_clean.txt')  # 清洗
# data_cut('wiki_corpus_clean.txt', 'wiki_corpus_cut.txt')  # 分词 一小时

start = time.time()

# vector_train('wiki_corpus_cut.txt', 'word2vec.model')  # 模型训练 
time1 = time.time()
print("time:", start - time1)

# word2vec_model('word2vec.model')  # 测试模型，和“说”相关的词语
time2 = time.time()
print("time:", time2 - time1)


# #############绘制关于“说” 的词云############

def draw_word_cloud(word_cloud):  # 绘制词云

    def get_mask():  # 获取一个圆形的mask
        x, y = np.ogrid[:300, :300]
        mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
        mask = 255 * mask.astype(int)
        return mask

    font_path = 'C:\Windows\Fonts\simhei.ttf'
    wc = WordCloud(background_color="white", mask=get_mask(), font_path=font_path)
    wc.generate_from_frequencies(word_cloud)

    # 隐藏x轴和y轴
    plt.axis("off")
    plt.imshow(wc, interpolation="bilinear")
    plt.show()


def test(model_path):
    model = word2vec.Word2Vec.load(model_path)
    one_corpus = ["问"]
    result = model.wv.most_similar(one_corpus[0], topn=100)

    word_cloud = dict()  # 将返回的结果转换为字典,便于绘制词云
    for sim in result:
        # print(sim[0],":",sim[1])
        word_cloud[sim[0]] = sim[1]

    draw_word_cloud(word_cloud)  # 绘制词云


test('word2vec.model')